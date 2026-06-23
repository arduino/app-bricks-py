# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import json
import os
import subprocess
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import download, download_and_extract, emit_json_error, write_manifest


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
    parser.add_argument(
        "--model-directory",
        default=os.environ.get("model_directory"),
        metavar="DIR",
        help="Name of the model sub-directory under --output-dir; used to "
             "write the manifest file (defaults to the 'model_directory' "
             "environment variable).",
    )

    args = parser.parse_args()

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
            artifact = download(url, args.output_dir, True)
            artifacts = [artifact]
        else:
            artifacts = download_and_extract(url, args.output_dir, True)
    except requests.HTTPError as exc:
        msg = f"HTTP error: {exc.response.status_code} {exc.response.reason}"
        emit_json_error(msg)
        sys.exit(1)
    except requests.RequestException as exc:
        msg = f"Request failed: {exc}"
        emit_json_error(msg)
        sys.exit(1)
    except Exception as exc:
        msg = f"Unexpected error: {exc}"
        emit_json_error(msg)
        sys.exit(1)

    # Write the download manifest as the final step so a partially-failed
    # download is never reported as complete by the checker.
    if not args.model_directory:
        emit_json_error(
            "Cannot write download manifest: --model-directory is not set "
            "and the 'model_directory' environment variable is empty."
        )
        sys.exit(1)
    try:
        manifest_dir = os.path.join(args.output_dir, args.model_directory)
        write_manifest(manifest_dir, artifacts)
    except OSError as exc:
        emit_json_error(f"Failed to write download manifest: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
