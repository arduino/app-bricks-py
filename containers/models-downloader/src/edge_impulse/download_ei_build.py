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
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import download, emit_json_error


BASE_URL = "https://studio.edgeimpulse.com/v1/api/{project_id}/deployment/download?type={target}&modelType={quantization}&impulseId={impulse_id}"


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
        required=True,
        default="float32",
        help="Quantization type of the model (e.g. float32, int8).",
    )
    parser.add_argument(
        "--target",
        required=True,
        default="runner-linux-aarch64",
        help="Target type of the model (e.g. runner-linux-aarch64, runner-linux-aarch64-qnn).",
    )

    args = parser.parse_args()

    url = BASE_URL.format(project_id=args.ei_project_id, impulse_id=args.impulse_id, quantization=args.quantization, target=args.target)

    try:
        out_file = download(url, args.output_dir, True, output_name=args.output_name)
        if os.path.isfile(out_file):
            os.chmod(out_file, 0o755)  # Ensure the file is executable
    except requests.HTTPError as exc:
        msg = f"HTTP error: {exc.response.status_code} {exc.response.reason}"
        emit_json_error(msg)
        sys.exit(1)
    except requests.RequestException as exc:
        msg = f"Request failed: {exc}"
        emit_json_error(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
