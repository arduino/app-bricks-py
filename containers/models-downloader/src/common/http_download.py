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


def emit_json_progress(event_type: str, description: str, current: int, total: int, unit: str, artifacts: list[str] | None = None):
    pct = round((current / total) * 100, 2) if total and total > 0 else 0
    data = {
        "event": event_type,
        "description": description,
        "current": current,
        "total": total,
        "unit": unit,
        "percentage": f"{pct}%",
    }
    if artifacts is not None:
        data["artifacts"] = artifacts
    print(json.dumps(data), flush=True)


def emit_json_error(description: str):
    data = {
        "event": "error",
        "description": description,
    }
    print(json.dumps(data), flush=True)


def check(url: str, output_name: str | None = None) -> dict[str, str | int | None]:
    """Perform a HEAD request on *url* and return content-length and filename.

    Args:
        url: URL to check.
        output_name: Optional filename override.  When ``None`` (or an empty
            string) the name is inferred from the ``Content-Disposition``
            header or the last path segment of *url*.

    Returns:
        A dict with keys ``filename`` and ``content_length``.
    """
    response = requests.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()

    if not output_name:
        output_name = None
    filename = output_name or _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
    content_length = int(response.headers.get("Content-Length", 0) or 0) or None

    return {"filename": filename, "content_length": content_length}


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
            emit_json_progress("start", f"Downloading {filename} from {url}", downloaded, total, "B")
            last_update = time.monotonic()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 1.0:
                        emit_json_progress("update", f"Downloading {filename} from {url}", downloaded, total, "B")
                        last_update = now
            emit_json_progress("complete", f"Downloaded {filename} from {url}", downloaded, total, "B", artifacts=[output_path])
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


def download_and_extract(url: str, output_dir: str, json_progress: bool, streaming: bool = True) -> list[str]:
    """Stream-download a ZIP from *url* and extract it to *output_dir*.

    Args:
        url: URL to download.
        output_dir: Directory where the ZIP contents will be extracted.
        json_progress: When ``True`` emit progress as JSON lines; otherwise
            use a ``tqdm`` progress bar (falling back to a simple inline bar
        streaming: When ``True`` (default), uses ``stream-unzip`` to decompress
            each entry as chunks arrive — no temporary file required and memory
            usage stays constant.  When ``False``, streams into a temporary file
            on disk first, then extracts with the stdlib ``zipfile`` module.

    Returns:
        List of paths (files only, no directories) of every entry extracted.
    """
    if streaming:
        return _download_and_extract_streaming(url, output_dir, json_progress)
    else:
        return _download_and_extract_buffered(url, output_dir, json_progress)


def _download_and_extract_streaming(url: str, output_dir: str, json_progress: bool) -> list[str]:
    from stream_unzip import stream_unzip

    os.makedirs(output_dir, exist_ok=True)
    extracted_files: list[str] = []
    pbar = None

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        filename = _filename_from_response(response, url.rstrip("/").split("/")[-1] or "download")
        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        last_update = time.monotonic()

        if json_progress:
            emit_json_progress("start", f"Downloading {filename} from {url}", 0, total, "B")
        else:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=filename)
            except ImportError:
                pass

        def byte_chunks():
            nonlocal downloaded, last_update
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if json_progress:
                    now = time.monotonic()
                    if now - last_update >= 1.0:
                        emit_json_progress("update", f"Downloading {filename} from {url}", downloaded, total, "B")
                        last_update = now
                elif pbar:
                    pbar.update(len(chunk))
                else:
                    print(f"\r{_simple_progress_bar(downloaded, total)}", end="", flush=True)
                yield chunk

        try:
            for zipped_path, _file_size, unzipped_chunks in stream_unzip(byte_chunks()):
                file_name = zipped_path.decode() if isinstance(zipped_path, bytes) else zipped_path
                output_path = os.path.join(output_dir, file_name)
                if file_name.endswith("/"):
                    os.makedirs(output_path, exist_ok=True)
                    for _ in unzipped_chunks:
                        pass
                else:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "wb") as f:
                        for chunk in unzipped_chunks:
                            f.write(chunk)
                    extracted_files.append(output_path)
        except Exception as exc:
            msg = f"Extraction failed: {exc}"
            if json_progress:
                emit_json_error(msg)
            else:
                print(msg, file=sys.stderr)
            raise
        finally:
            if pbar:
                pbar.close()
            elif not json_progress:
                print()

    if json_progress:
        print(json.dumps({"event": "complete", "description": f"Extracted to: {output_dir}", "artifacts": extracted_files}), flush=True)
    else:
        print(f"Extracted to: {output_dir}")

    return extracted_files


def _download_and_extract_buffered(url: str, output_dir: str, json_progress: bool) -> list[str]:
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
                    emit_json_progress("start", f"Downloading {filename} from {url}", downloaded, total, "B")
                    last_update = time.monotonic()
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_update >= 1.0:
                            emit_json_progress("update", f"Downloading {filename} from {url}", downloaded, total, "B")
                            last_update = now
                    emit_json_progress("complete", f"Downloaded {filename} from {url}", downloaded, total, "B")
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
            print(json.dumps({"event": "info", "description": f"Extracting {filename} to {output_dir}"}), flush=True)
        else:
            print(f"Extracting {filename} to {output_dir}")

        extracted_artifacts: list[str] = []
        try:
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(output_dir)
                extracted_artifacts = [
                    os.path.join(output_dir, info.filename)
                    for info in zf.infolist()
                    if not info.is_dir()
                ]
        except (OSError, zipfile.BadZipFile) as exc:
            msg = f"Extraction failed: {exc}"
            if json_progress:
                emit_json_error(msg)
            else:
                print(msg, file=sys.stderr)
            raise

        if json_progress:
            print(json.dumps({"event": "complete", "description": f"Extracted to: {output_dir}", "artifacts": extracted_artifacts}), flush=True)
        else:
            print(f"Extracted to: {output_dir}")

        return extracted_artifacts
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


MANIFEST_FILENAME = "downloaded.json"


def write_manifest(
    directory: str,
    files: list[str],
    model_id: str,
    manifest_name: str = MANIFEST_FILENAME,
) -> str:
    """Write a manifest describing every file that belongs to a download.

    The manifest is a JSON document of the form::

        {
            "version": 1,
            "model_id": "<id from models-list.yaml>",
            "files": [
                {"path": "<path relative to *directory*>", "size": <bytes>},
                ...
            ]
        }

    Storing relative paths keeps the manifest valid even if the model
    directory is moved or bind-mounted at a different location. The
    ``model_id`` lets the consumer attribute each manifest back to its
    model regardless of where the artifacts live on disk.

    Args:
        directory: Directory the manifest is written into (and that all
            *files* must live under).
        files: Absolute paths of every file produced by the download.
        model_id: ID of the model this download belongs to, as declared
            in models-list.yaml.
        manifest_name: Filename of the manifest within *directory*.  The
            default of ``downloaded.json`` is intended for downloads whose
            artifacts live in a dedicated model directory; downloads that
            place several independent artifacts side-by-side (e.g. one
            ``.eim`` file per Edge Impulse model, or multiple Hugging Face
            quantizations of the same repo) should pass a per-artifact name
            to avoid clobbering each other's manifest.

    Returns:
        The absolute path of the manifest file that was written.
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)

    entries: list[dict[str, object]] = []
    for path in files:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, directory)
        entries.append({"path": rel_path, "size": os.path.getsize(abs_path)})

    manifest_path = os.path.join(directory, manifest_name)
    # Write atomically so a crash mid-write never leaves a half-valid manifest.
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"version": 1, "model_id": model_id, "files": entries}, f)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def verify_manifest(directory: str, manifest_name: str = MANIFEST_FILENAME) -> tuple[bool, str]:
    """Verify the manifest in *directory* against the files on disk.

    Returns:
        ``(True, "")`` if the manifest exists and every listed file is
        present with the expected size; ``(False, <reason>)`` otherwise.
    """
    manifest_path = os.path.join(directory, manifest_name)
    if not os.path.isfile(manifest_path):
        return False, f"manifest missing: {manifest_path}"

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"manifest unreadable: {exc}"

    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list) or not files:
        return False, "manifest contains no files"

    for entry in files:
        rel_path = entry.get("path") if isinstance(entry, dict) else None
        expected_size = entry.get("size") if isinstance(entry, dict) else None
        if not isinstance(rel_path, str) or not isinstance(expected_size, int):
            return False, f"manifest entry is malformed: {entry!r}"
        abs_path = os.path.join(directory, rel_path)
        if not os.path.isfile(abs_path):
            return False, f"file missing: {rel_path}"
        actual_size = os.path.getsize(abs_path)
        if actual_size != expected_size:
            return False, (
                f"size mismatch for {rel_path}: expected {expected_size} bytes, "
                f"found {actual_size} bytes"
            )

    return True, ""


def is_complete(directory: str, manifest_name: str = MANIFEST_FILENAME) -> bool:
    """Return ``True`` if ``<directory>/<manifest_name>`` exists and verifies.

    Convenience wrapper around :func:`verify_manifest` for callers that only
    need a boolean. The contract used by every handler is:

    * manifest present and verified → previous download is complete, skip it;
    * manifest absent or invalid    → previous download is incomplete, the
      caller must wipe whatever leftovers it owns and download again.
    """
    ok, _ = verify_manifest(directory, manifest_name=manifest_name)
    return ok
