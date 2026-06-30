# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import json
import os
import shutil
import subprocess
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.download_marker import write_marker
from common.http_download import download, download_and_extract, emit_json_error, install_signal_handlers


def _wipe_model_dir(model_dir: str, base_dir: str) -> None:
    """Remove the partial model directory (and its ``.download`` marker) after a
    failed or interrupted download. Never removes *base_dir* itself, which is the
    mounted ``/models`` directory and cannot be deleted from inside the container.
    """
    if os.path.abspath(model_dir) == os.path.abspath(base_dir):
        return
    shutil.rmtree(model_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download an AI Hub model via the AI Hub API.")
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        metavar="TYPE",
        help="AI Hub model type (e.g. voice_ai).",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        metavar="NAME",
        help="AI Hub model name (e.g. melotts_zh).",
    )
    parser.add_argument(
        "--quantization",
        required=True,
        type=str,
        metavar="QUANTIZATION",
        help="Quantization type of the model (e.g. float32, int8, mixed_with_float).",
    )
    parser.add_argument(
        "--chipset",
        required=True,
        type=str,
        metavar="CHIPSET",
        help="Chipset type of the model (e.g. qualcomm-qcs8275).",
    )
    parser.add_argument(
        "--version",
        type=str,
        metavar="VERSION",
        help="Version of the model (e.g. 0.51.0).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to save the downloaded file (default: current directory).",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Save the raw .zip file instead of extracting its contents (default: extract in-memory during download).",
    )

    args = parser.parse_args()

    # Ensure SIGINT/SIGTERM (e.g. `docker stop`) trigger cleanup of partial
    # downloads/extractions before exiting. SIGKILL (-9) cannot be caught.
    install_signal_handlers()

    # In-progress marker shared with the listing tool; it lives *inside* the
    # model directory and is written before the (interruptible) fetch so an
    # aborted run is never mistaken for an installed model. On success only the
    # marker is cleared; on interrupt or error the whole model directory (marker
    # + partial files) is removed so the next run starts fresh and the listing
    # tool never sees a phantom (empty) model directory.
    model_dir = os.path.join(args.output_dir, os.environ.get("model_directory", ""))
    marker = write_marker(
        model_dir,
        handler="ai-hub-handler",
        models_repository=os.environ.get("models_repository", ""),
        model_directory=os.environ.get("model_directory", ""),
    )

    # Build the qai_hub_models fetch command to retrieve the download URL.
    # model_name, model_type, quantization and chipset are mandatory;
    # version is optional.
    cmd = [
        "qai_hub_models",
        "fetch",
        args.model_name,
        "-r",
        args.model_type,
        "-p",
        args.quantization,
        "-c",
        args.chipset,
    ]
    if args.version:
        cmd += ["-v", args.version]
    cmd.append("--url-only")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        if not url or url == "" or not url.startswith("http"):
            raise ValueError("Received wrong URL from qai_hub_models fetch command: " + url)
    except subprocess.CalledProcessError as exc:
        msg = f"Failed to fetch model URL: {exc.stderr.strip() or exc}"
        emit_json_error(msg)
        sys.exit(1)

    print(json.dumps({"event": "info", "description": f"Downloading model from: {url}"}), flush=True)

    try:
        if args.no_unzip:
            download(url, args.output_dir, True)
        else:
            download_and_extract(url, args.output_dir, True)
        # Download finished successfully: clear the in-progress marker.
        if os.path.exists(marker):
            os.remove(marker)
    except requests.HTTPError as exc:
        msg = f"HTTP error: {exc.response.status_code} {exc.response.reason}"
        _wipe_model_dir(model_dir, args.output_dir)
        emit_json_error(msg)
        sys.exit(1)
    except requests.RequestException as exc:
        msg = f"Request failed: {exc}"
        _wipe_model_dir(model_dir, args.output_dir)
        emit_json_error(msg)
        sys.exit(1)
    except KeyboardInterrupt:
        _wipe_model_dir(model_dir, args.output_dir)
        emit_json_error("Download interrupted by signal; partial files removed")
        sys.exit(130)
    except Exception as exc:
        msg = f"Unexpected error: {exc}"
        _wipe_model_dir(model_dir, args.output_dir)
        emit_json_error(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
