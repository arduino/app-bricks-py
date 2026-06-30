# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""
hf_downloader — Hugging Face Model Downloader CLI

A command-line tool for downloading GGUF-format models from Hugging Face
repositories. It targets llama.cpp-style repos that may contain multiple
quantization variants and optional multimodal projection (mmproj) files.
After downloading, it auto-generates a ``models.ini`` configuration file
that indexes all downloaded models for use by downstream runners.

Usage — three modes
--------------------
1. Compact key::

       hf_downloader --model-key llamacpp:<repo_id>:<quantization>[:<mmproj_quantization>]

2. Explicit names::

       hf_downloader --model-repo-id <repo_id> --model-name <file> [--model-mmproj-name <file>]

3. Direct URL::

       hf_downloader --model-url https://huggingface.co/<org>/<repo>/resolve/main/<file>

Key options
-----------
--output-dir DIR        Destination directory (default: current directory).
                        Files are saved under ``<output-dir>/<repo-id>/``.
--hf-token KEY          Hugging Face API token for gated/private repositories.
--verbose               Print resolved parameters before downloading.

After all files are downloaded, ``models.ini`` is written to ``<output-dir>``
mapping each model stem to its GGUF path (and mmproj path where present).
"""

import fnmatch
import os
import re
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile
import argparse
import configparser
from pathlib import Path
from tqdm.auto import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.download_marker import write_marker


def emit_json_info(description: str, artifacts: list[str] | None = None, downloading: bool | None = None):
    data: dict = {"event": "info", "description": description}
    if artifacts is not None:
        data["artifacts"] = artifacts
    if downloading is not None:
        data["downloading"] = downloading
    print(json.dumps(data), flush=True)


def emit_json_error(description: str, downloading: bool | None = None):
    data: dict = {"event": "error", "description": description}
    if downloading is not None:
        data["downloading"] = downloading
    print(json.dumps(data), flush=True)


def remove_model_dir(output_dir: str, base_dir: str) -> None:
    """Remove the repo directory and prune now-empty parent dirs up to base_dir.

    repo_id may contain a '/', so output_dir is nested (e.g.
    <base>/moondream/moondream2-gguf). Deleting only output_dir would leave an
    empty org directory (<base>/moondream) behind; walk up removing empty parents,
    stopping at base_dir (the mounted /models, which is never removed).
    """
    base = os.path.abspath(base_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    parent = os.path.dirname(os.path.abspath(output_dir))
    while parent != base and parent.startswith(base + os.sep):
        try:
            os.rmdir(parent)  # only succeeds if the directory is empty
        except OSError:
            break
        parent = os.path.dirname(parent)


def install_signal_handlers() -> None:
    """Translate SIGINT/SIGTERM into KeyboardInterrupt so cleanup runs before
    exit. SIGKILL (-9) cannot be caught."""
    import signal

    def _handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class JsonProgress(tqdm):
    # Minimum seconds between emitted "update" events to avoid flooding stdout.
    EMIT_INTERVAL = 1.0

    def __init__(self, *args, **kwargs):
        self._complete_emitted = False
        self._last_emit = 0.0
        super().__init__(*args, **kwargs)
        # Emit an initial "start" event
        self._emit("start")

    def _emit(self, event_type):
        """Helper to print the current state as JSON"""
        self._last_emit = time.monotonic()
        pct = round((self.n / self.total) * 100, 2) if self.total else 0
        data = {
            "event": event_type,
            "description": self.desc,
            "current": self.n,
            "total": self.total,
            "unit": self.unit,
            "percentage": f"{pct}%",
        }
        print(json.dumps(data), flush=True)

    def update(self, n=1):
        displayed = super().update(n)
        # Throttle: only emit an "update" event once EMIT_INTERVAL has elapsed.
        if time.monotonic() - self._last_emit >= self.EMIT_INTERVAL:
            self._emit("update")
        return displayed

    def close(self):
        # Only report completion if the transfer actually finished.
        if self.total and self.n >= self.total and not self._complete_emitted:
            self._complete_emitted = True
            self._emit("complete")
        super().close()

    def display(self, msg=None, pos=None):
        # Do not display the progress bar in the terminal, we will emit JSON events instead
        pass


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """Parse a Hugging Face URL and return (repo_id, filename, revision).

    Supports URLs like:
      https://huggingface.co/<org>/<repo>/resolve/<revision>/<filename>
      https://huggingface.co/<org>/<repo>/blob/<revision>/<filename>
    """
    match = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+?)(?:\?.*)?$",
        url,
    )
    if not match:
        raise ValueError(f"Invalid Hugging Face URL: {url}\nExpected format: https://huggingface.co/<org>/<repo>/resolve/<revision>/<filename>")
    repo_id = match.group(1)
    revision = match.group(2)
    filename = match.group(3)
    return repo_id, filename, revision


def delete_matched_files(output_dir: str, models_base: str, allow_pattern: str, verbose: bool = False):
    """Delete files inside output_dir whose names match allow_pattern (fnmatch-style).
    After deletion, removes any empty subdirectories but never output_dir itself.
    """
    base = Path(output_dir)
    models_base_path = Path(models_base)
    if not base.exists():
        emit_json_info(f"Directory does not exist, nothing to delete: {output_dir}")
        return
    matched = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, allow_pattern)]
    if not matched:
        emit_json_info(f"No files matching '{allow_pattern}' found in {output_dir}")
        return
    dirs_to_check: set[Path] = set()
    for f in matched:
        if verbose:
            emit_json_info(f"Deleting: {f}")
        dirs_to_check.add(f.parent)
        f.unlink()
    # Remove empty subdirectories (deepest first), but never output_dir itself
    for d in sorted(dirs_to_check, key=lambda p: len(p.parts), reverse=True):
        if d == models_base:
            continue
        if d.exists() and not any(d.iterdir()):
            if verbose:
                emit_json_info(f"Removing empty directory: {d}")
            d.rmdir()
    # Remove all empty directories up to output_dir. List all directories under models_base and check if they are empty, removing them
    for d in sorted(models_base_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and d != base and not any(d.iterdir()):
            if verbose:
                emit_json_info(f"Removing empty directory: {d}")
            d.rmdir()


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()

    for gguf_file in sorted(models_dir.rglob("*.gguf")):
        if "mmproj" in gguf_file.name:
            continue

        section = gguf_file.stem
        config[section] = {}
        config[section]["model"] = str(gguf_file.as_posix())

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            config[section]["mmproj"] = str(mmproj_files[0].as_posix())

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    emit_json_info(f"Generated models.ini with {len(config.sections())} model(s)", artifacts=[str(output_path)])


def main():
    parser = argparse.ArgumentParser(description="Download an Hugging Face model via HF download API")
    parser.add_argument(
        "--model-key",
        type=str,
        metavar="KEY",
        help="model key (e.g. llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16). "
        "The format is: <model_type>:<repo_id>:<quantization>:<optional mmproj quantization>.",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        metavar="URL",
        help="Direct Hugging Face file URL (e.g. https://huggingface.co/org/repo/resolve/main/model.gguf). "
        "Supports both /resolve/ and /blob/ URL formats.",
    )
    parser.add_argument(
        "--model-mmproj-url",
        type=str,
        metavar="URL",
        help="Direct Hugging Face URL for the mmproj file (e.g. https://huggingface.co/org/repo/resolve/main/mmproj-BF16.gguf). "
        "Only used with --model-url.",
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        metavar="KEY",
        help="model repository ID (e.g. llamacpp:unsloth/gemma-4-E4B-it-GGUF). Only used if --model-key is not provided.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        metavar="KEY",
        help="model name (e.g. gemma-4-E2B-it-Q4_0.gguf). Only used if --model-key is not provided.",
    )
    parser.add_argument(
        "--model-mmproj-name",
        type=str,
        metavar="KEY",
        help="model mmproj name (e.g. mmproj-F16.gguf). Only used if --model-key is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to save the downloaded file (default: current directory).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        metavar="KEY",
        help="Hugging Face API token for authentication.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete already-present files matching the resolved patterns instead of downloading them.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if model files matching the resolved patterns are present on the filesystem.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print the total size (in bytes) of files matching the resolved patterns on Hugging Face.",
    )

    args = parser.parse_args()

    allow_pattern = None
    mmproj_allow_pattern = None
    url_filename = None  # set when --model-url is used (single-file download)
    url_revision = None
    mmproj_url_filename = None
    mmproj_url_revision = None

    if args.model_url and args.model_url != "":
        repo_id, url_filename, url_revision = parse_hf_url(args.model_url)
        allow_pattern = url_filename.split("/")[-1]  # use basename as pattern for check/delete

        if args.model_mmproj_url and args.model_mmproj_url != "":
            _, mmproj_url_filename, mmproj_url_revision = parse_hf_url(args.model_mmproj_url)
            mmproj_allow_pattern = mmproj_url_filename.split("/")[-1]

        if args.verbose:
            emit_json_info(f"Parsed URL — Repository ID: {repo_id}")
            emit_json_info(f"Filename: {url_filename}")
            emit_json_info(f"Revision: {url_revision}")
            if mmproj_url_filename:
                emit_json_info(f"MMProj Filename: {mmproj_url_filename}")
                emit_json_info(f"MMProj Revision: {mmproj_url_revision}")

    elif args.model_key and args.model_key != "":
        model_type, repo_id, quantization, *mmproj_quantization = args.model_key.split(":")
        if repo_id == "":
            raise ValueError("repo_id cannot be empty")
        if quantization == "":
            raise ValueError("quantization cannot be empty")

        if args.verbose:
            emit_json_info(f"Repository ID: {repo_id}")
            emit_json_info(f"Model key: {args.model_key}")
            emit_json_info(f"Model type: {model_type}")
            emit_json_info(f"Quantization: {quantization}")
            if mmproj_quantization:
                emit_json_info(f"MMProj Quantization: {mmproj_quantization[0]}")

        allow_pattern = f"*{quantization}*.gguf"
        mmproj_allow_pattern = f"*mmproj*{mmproj_quantization[0]}*.gguf" if mmproj_quantization else None
    else:
        if not args.model_repo_id or not args.model_name:
            raise ValueError("If --model-key is not provided, both --model-repo-id and --model-name must be specified")

        repo_id = args.model_repo_id

        allow_pattern = args.model_name
        if allow_pattern == "":
            raise ValueError("model name cannot be empty")
        if "*" not in allow_pattern and not allow_pattern.endswith(".gguf"):
            allow_pattern = f"*{allow_pattern}*"

        if args.model_mmproj_name and args.model_mmproj_name != "":
            mmproj_allow_pattern = args.model_mmproj_name
            if "*" not in mmproj_allow_pattern and not mmproj_allow_pattern.endswith(".gguf"):
                mmproj_allow_pattern = f"*{mmproj_allow_pattern}*"

        if args.verbose:
            emit_json_info(f"Repository ID: {repo_id}")
            emit_json_info(f"Model identifier: {allow_pattern}")
            if mmproj_allow_pattern:
                emit_json_info(f"MMProj file: {mmproj_allow_pattern}")

    if args.hf_token and args.hf_token != "":
        os.environ["HF_HUB_TOKEN"] = args.hf_token

    # Create download folder if it doesn't exist. Patter is: output_dir + / repo_id
    output_dir = f"{args.output_dir}/{repo_id}"

    if args.info:
        patterns = [allow_pattern]
        if mmproj_allow_pattern:
            patterns.append(mmproj_allow_pattern)
        api = HfApi()
        all_files = [item for item in api.list_repo_tree(repo_id=repo_id, recursive=True) if isinstance(item, RepoFile)]
        matched_files = [
            {"file": f.path, "size": f.size} for f in all_files if f.size and any(fnmatch.fnmatch(f.path.split("/")[-1], p) for p in patterns)
        ]
        total_bytes = sum(f["size"] for f in matched_files)
        print(
            json.dumps({
                "event": "stat",
                "description": f"Total download size for {repo_id}",
                "size_bytes": total_bytes,
                "size_mb": round(total_bytes / 1024 / 1024, 2),
                "files": matched_files,
            }),
            flush=True,
        )
    elif args.check:
        base = Path(output_dir)
        # A ".download" marker means a download is in progress or was interrupted
        if (base / ".download").is_file():
            emit_json_info(f"Model downloading: {repo_id}", downloading=True)
        else:
            matched = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, allow_pattern)] if base.exists() else []
            if mmproj_allow_pattern:
                matched += [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, mmproj_allow_pattern)] if base.exists() else []
            if matched:
                emit_json_info(f"Model exists: {allow_pattern}", downloading=False)
            else:
                emit_json_error(f"Model does not exist: {allow_pattern}", downloading=False)
                raise SystemExit(1)
    elif args.delete:
        if args.verbose:
            emit_json_info(f"Deleting files matching '{allow_pattern}' in {output_dir}")
        delete_matched_files(output_dir, args.output_dir, allow_pattern, args.verbose)
        if mmproj_allow_pattern:
            if args.verbose:
                emit_json_info(f"Deleting mmproj files matching '{mmproj_allow_pattern}' in {output_dir}")
            delete_matched_files(output_dir, args.output_dir, mmproj_allow_pattern, args.verbose)

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))
    else:
        # Per-repo ".download" marker: present => prior run killed mid-download,
        # wipe and retry; absent but dir exists => already complete.
        marker = Path(output_dir) / ".download"
        if marker.is_file():
            emit_json_info(f"Removing incomplete previous download: {repo_id}")
            remove_model_dir(output_dir, args.output_dir)
        elif os.path.isdir(output_dir):
            emit_json_info(f"Model exists: {repo_id}")
            return

        os.makedirs(output_dir, exist_ok=True)
        write_marker(
            output_dir,
            handler="hf-handler",
            models_repository=os.environ.get("models_repository", ""),
            model_directory=os.environ.get("model_directory", "") or repo_id,
        )

        emit_json_info(f"Downloading to: {os.path.abspath(output_dir)}", artifacts=[os.path.abspath(output_dir)])

        tqdm_class = JsonProgress

        try:
            if url_filename:
                # Single-file download via direct URL
                if args.verbose:
                    emit_json_info(f"Downloading file '{url_filename}' from {repo_id} (revision: {url_revision})")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=url_filename,
                    revision=url_revision,
                    local_dir=output_dir,
                    tqdm_class=tqdm_class,
                )
                if mmproj_url_filename:
                    if args.verbose:
                        emit_json_info(f"Downloading mmproj file '{mmproj_url_filename}' from {repo_id} (revision: {mmproj_url_revision})")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=mmproj_url_filename,
                        revision=mmproj_url_revision,
                        local_dir=output_dir,
                        tqdm_class=tqdm_class,
                    )
            else:
                # Pattern-based download via snapshot
                if args.verbose:
                    emit_json_info(f"Downloading model from Hugging Face repository: {repo_id} with allow pattern: {allow_pattern}")
                snapshot_download(
                    repo_id=repo_id, allow_patterns=[allow_pattern], ignore_patterns=["*mmproj*"], local_dir=output_dir, tqdm_class=tqdm_class
                )

                if mmproj_allow_pattern:
                    if args.verbose:
                        emit_json_info(
                            f"Downloading mmproj model file from Hugging Face repository: {repo_id} with allow pattern: {mmproj_allow_pattern}"
                        )
                    snapshot_download(repo_id=repo_id, allow_patterns=[mmproj_allow_pattern], local_dir=output_dir, tqdm_class=tqdm_class)
        except BaseException:
            # Network/extraction errors and SIGINT/SIGTERM-driven KeyboardInterrupt
            # leave a partial repo directory; remove it before exiting.
            if os.path.isdir(output_dir):
                remove_model_dir(output_dir, args.output_dir)
            raise

        # Remove download caches
        cache_path = Path(output_dir) / ".cache"
        if cache_path.is_dir():
            shutil.rmtree(cache_path)

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))

        # Report the absolute path(s) of the downloaded model file(s).
        downloaded = sorted(str(p.resolve()) for p in Path(output_dir).rglob("*.gguf"))
        emit_json_info(f"Downloaded to: {output_dir}", artifacts=downloaded)

        # Download finished successfully: clear the in-progress marker.
        marker = Path(output_dir) / ".download"
        if marker.exists():
            marker.unlink()


if __name__ == "__main__":
    install_signal_handlers()
    try:
        main()
    except KeyboardInterrupt:
        emit_json_error("Download interrupted by signal; partial files removed")
        raise SystemExit(130)
