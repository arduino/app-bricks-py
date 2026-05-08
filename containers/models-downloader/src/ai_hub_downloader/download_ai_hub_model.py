# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

import requests


CHUNK_SIZE = 1024 * 1024  # 1 MB


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


def emit_json_progress(event_type: str, description: str, current: int, total: int, unit: str):
    pct = round((current / total) * 100, 2) if total else 0
    data = {
        "event": event_type,
        "description": description,
        "current": current,
        "total": total,
        "unit": unit,
        "percentage": f"{pct}%",
    }
    print(json.dumps(data), flush=True)


def download(url: str, output_dir: str, json_progress: bool):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        filename = _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0

        if json_progress:
            emit_json_progress("start", filename, downloaded, total, "B")
            last_update = time.monotonic()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 1.0:
                        emit_json_progress("update", filename, downloaded, total, "B")
                        last_update = now
            emit_json_progress("complete", filename, downloaded, total, "B")
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
        "--json-progress",
        action="store_true",
        help='Report progress as JSON lines, e.g. {"progress": "42%%"}, instead of a progress bar.',
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not automatically unzip the downloaded file (default: unzip if the file is a .zip archive).",
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
        if args.json_progress:
            print(json.dumps({"error": msg}), flush=True)
        else:
            print(msg, file=sys.stderr)
        sys.exit(1)

    try:
        if args.json_progress:
            print(json.dumps({"event": "info", "description": f"Downloading model from: {url}"}), flush=True)
        else:
            print(f"Downloading model from: {url}")
        output_path = download(url, args.output_dir, args.json_progress)
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

    if not args.no_unzip and output_path.lower().endswith(".zip"):
        if args.json_progress:
            print(json.dumps({"event": "info", "description": f"Unzipping: {output_path}"}), flush=True)
        else:
            print(f"Unzipping: {output_path}")
        try:
            with zipfile.ZipFile(output_path, "r") as zf:
                zf.extractall(args.output_dir)
            if args.json_progress:
                print(json.dumps({"event": "info", "description": f"Extracted to: {args.output_dir}"}), flush=True)
            else:
                print(f"Extracted to: {args.output_dir}")
        except Exception as exc:
            msg = f"Failed to unzip {output_path}: {exc}"
            if args.json_progress:
                print(json.dumps({"error": msg}), flush=True)
            else:
                print(msg, file=sys.stderr)
            sys.exit(1)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    main()
