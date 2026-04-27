# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""Download an Edge Impulse deployment build artifact.

Usage examples:
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --output-dir ./downloads
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --json-progress
    python download_ei_build.py --ei-project-id 948887 --impulse-id 11 --api-key <key>
"""

import argparse
import json
import os
import sys

import requests


BASE_URL = "https://studio.edgeimpulse.com/v1/api/{project_id}/deployment/download?impulseId={impulse_id}&type=runner-linux-aarch64-qnn&engine=tflite"
CHUNK_SIZE = 8192


def _filename_from_response(response: requests.Response, fallback: str) -> str:
    cd = response.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        return cd.split("filename=")[-1].strip().strip('"').strip("'")
    return fallback


def _simple_progress_bar(downloaded: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"{downloaded} B"
    pct = downloaded / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct * 100:.1f}%  ({downloaded}/{total} B)"


def download(url: str, output_dir: str, output_name: str | None, json_progress: bool):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        if output_name is not None and output_name == "":
            output_name = None
        filename = output_name or _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0

        if json_progress:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded / total * 100) if total > 0 else -1
                    record = {"progress": f"{pct}%"} if pct >= 0 else {"downloaded_bytes": downloaded}
                    print(json.dumps(record), flush=True)
            print(json.dumps({"progress": "100%", "file": output_path}), flush=True)
        else:
            try:
                from tqdm import tqdm

                with tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
            except ImportError:
                # Fallback: simple inline progress bar without tqdm
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r{_simple_progress_bar(downloaded, total)}", end="", flush=True)
                print()

            print(f"Saved to: {output_path}")

    return output_path


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
        metavar="FILE",
        help="Name of the downloaded file.",
    )
    parser.add_argument(
        "--json-progress",
        action="store_true",
        help='Report progress as JSON lines, e.g. {"progress": "42%%"}, instead of a progress bar.',
    )

    args = parser.parse_args()

    url = BASE_URL.format(project_id=args.ei_project_id, impulse_id=args.impulse_id)

    try:
        print(f"Downloading from: {url}")
        download(url, args.output_dir, args.output_name, args.json_progress)
    except requests.HTTPError as exc:
        msg = f"HTTP error: {exc.response.status_code} {exc.response.reason}"
        if args.json_progress:
            print(json.dumps({"error": msg}), flush=True)
        else:
            print(msg, file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        msg = f"Request failed: {exc}"
        if args.json_progress:
            print(json.dumps({"error": msg}), flush=True)
        else:
            print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
