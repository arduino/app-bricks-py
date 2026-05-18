# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Shared HTTP download utilities used by multiple model downloaders."""

import json
import os
import sys
import tempfile
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


def emit_json_error(description: str):
    data = {
        "event": "error",
        "description": description,
    }
    print(json.dumps(data), flush=True)


def download(url: str, output_dir: str, json_progress: bool, output_name: str | None = None) -> str:
    """Download *url* to *output_dir* and return the local file path.

    Args:
        url: URL to download.
        output_dir: Directory where the file will be saved.
        json_progress: When ``True`` emit progress as JSON lines; otherwise
            use a ``tqdm`` progress bar (falling back to a simple inline bar
            if ``tqdm`` is not installed).
        output_name: Optional filename override.  When ``None`` (or an empty
            string) the name is inferred from the ``Content-Disposition``
            header or the last path segment of *url*.
    """
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        if not output_name:
            output_name = None
        filename = output_name or _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
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


def download_and_extract(url: str, output_dir: str, json_progress: bool) -> None:
    """Stream-download a ZIP from *url* and extract it to *output_dir*.

    The response is streamed chunk-by-chunk into a temporary file on disk
    (so arbitrarily large archives never reside fully in memory).  Once the
    download is complete the archive is extracted and the temporary file is
    deleted.
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = None
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            filename = _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
            total = int(response.headers.get("Content-Length", 0) or 0)
            downloaded = 0

            with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name

                if json_progress:
                    emit_json_progress("start", filename, downloaded, total, "B")
                    last_update = time.monotonic()
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        tmp.write(chunk)
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
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                tmp.write(chunk)
                                downloaded += len(chunk)
                                pbar.update(len(chunk))
                    except ImportError:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            tmp.write(chunk)
                            downloaded += len(chunk)
                            print(f"\r{_simple_progress_bar(downloaded, total)}", end="", flush=True)
                        print()

        if json_progress:
            print(json.dumps({"event": "info", "description": f"Extracting: {filename}"}), flush=True)
        else:
            print(f"Extracting: {filename}")

        try:
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(output_dir)
        except (OSError, zipfile.BadZipFile) as exc:
            msg = f"Extraction failed: {exc}"
            if json_progress:
                emit_json_error(msg)
            else:
                print(msg, file=sys.stderr)
            raise

        if json_progress:
            print(json.dumps({"event": "info", "description": f"Extracted to: {output_dir}"}), flush=True)
        else:
            print(f"Extracted to: {output_dir}")
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
