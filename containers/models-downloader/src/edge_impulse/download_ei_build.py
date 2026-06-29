# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Download an Edge Impulse deployment build artifact.

Usage examples:
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --output-name model.eim --output-dir ./downloads \
        --quantization int8 --target runner-linux-aarch64-qnn
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --output-name model.eim --output-dir ./downloads
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --output-name model.eim
"""

import argparse
import os
import shutil
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import check, download, emit_json_error, install_signal_handlers


BASE_URL = "https://studio.edgeimpulse.com/v1/api/{project_id}/deployment/download?type={target}&impulseId={impulse_id}"


def _wipe_model_dir(model_dir: str) -> None:
    """Remove the partial model directory (and its ``.download`` marker) after a
    failed or interrupted download. Refuses to remove the top-level ``/models``
    mount, which cannot be deleted from inside the container.
    """
    abs_dir = os.path.abspath(model_dir)
    if abs_dir in (os.path.abspath("/models"), os.sep):
        return
    shutil.rmtree(model_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download an Edge Impulse deployment build artifact via the EI REST API.")
    parser.add_argument(
        "--ei-project-id",
        required=True,
        type=int,
        metavar="ID",
        help="Edge Impulse project ID (e.g. 948887).",
    )
    parser.add_argument(
        "--impulse-id",
        required=True,
        type=int,
        metavar="N",
        help="Impulse ID (e.g. 11).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to save the downloaded file (default: current directory).",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        metavar="FILE",
        help="Name of the downloaded file.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Quantization type of the model (e.g. float32, int8). If omitted, not sent as a query parameter.",
    )
    parser.add_argument(
        "--target",
        required=True,
        default="runner-linux-aarch64",
        help="Target type of the model (e.g. runner-linux-aarch64, runner-linux-aarch64-qnn).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Only retrieve file size and name via a HEAD request (no download).",
    )

    args = parser.parse_args()

    # Ensure SIGINT/SIGTERM (e.g. `docker stop`) trigger cleanup of partial
    # downloads before exiting. SIGKILL (-9) cannot be caught.
    install_signal_handlers()

    url = BASE_URL.format(project_id=args.ei_project_id, impulse_id=args.impulse_id, target=args.target)
    if args.quantization:
        url += f"&modelType={args.quantization}"

    # In-progress marker shared with the listing tool; it lives *inside* the
    # model folder (mirrors AI Hub / HF). On success only the marker is cleared;
    # on interrupt or error the whole model folder (marker + partial files) is
    # removed so the next run starts fresh and the listing tool never sees a
    # phantom (empty) model folder that would be mistaken for an installed model.
    marker = os.path.join(args.output_dir, ".download")

    try:
        if args.info:
            import json

            info = check(url, output_name=args.output_name)
            print(
                json.dumps({
                    "event": "stat",
                    "description": f"Model info for project {args.ei_project_id} impulse {args.impulse_id}",
                    "filename": info["filename"],
                    "size_bytes": info["content_length"],
                    "size_mb": round(info["content_length"] / 1024 / 1024, 2) if info["content_length"] else None,
                }),
                flush=True,
            )
        else:
            out_file = download(url, args.output_dir, True, output_name=args.output_name)
            if os.path.isfile(out_file):
                os.chmod(out_file, 0o755)  # Ensure the file is executable
            # Download finished successfully: clear the in-progress marker.
            if os.path.exists(marker):
                os.remove(marker)
    except requests.HTTPError as exc:
        msg = f"HTTP error: {exc.response.status_code} {exc.response.reason} (url: {url})"
        if not args.info:
            _wipe_model_dir(args.output_dir)
        emit_json_error(msg)
        sys.exit(1)
    except requests.RequestException as exc:
        msg = f"Request failed: {exc} (url: {url})"
        if not args.info:
            _wipe_model_dir(args.output_dir)
        emit_json_error(msg)
        sys.exit(1)
    except KeyboardInterrupt:
        if not args.info:
            _wipe_model_dir(args.output_dir)
        emit_json_error("Download interrupted by signal; partial files removed")
        sys.exit(130)


if __name__ == "__main__":
    main()
