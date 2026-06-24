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

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile
import argparse
import configparser
from pathlib import Path
from tqdm.auto import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import is_complete, write_manifest


def emit_json_info(description: str, artifacts: list[str] | None = None):
    data: dict = {"event": "info", "description": description}
    if artifacts is not None:
        data["artifacts"] = artifacts
    print(json.dumps(data), flush=True)


def emit_json_error(description: str):
    print(json.dumps({"event": "error", "description": description}), flush=True)


class JsonProgress(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Emit an initial "start" event
        self._emit("start")

    def _emit(self, event_type):
        """Helper to print the current state as JSON"""
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
        self._emit("update")
        return displayed

    def close(self):
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


def _matched_files(base: Path, pattern: str) -> list[Path]:
    if not base.exists():
        return []
    return [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, pattern)]


def _manifest_name(allow_pattern: str, mmproj_allow_pattern: str | None) -> str:
    """Build a deterministic, filesystem-safe manifest filename for a (pattern,
    mmproj_pattern) request. Different requests against the same repository
    therefore get independent manifests and never overwrite each other.

    The name is suffixed with ``.downloaded.json`` so that a single
    ``*downloaded.json`` glob covers every handler's manifest naming
    convention (AI Hub, Edge Impulse, Hugging Face).
    """
    raw = allow_pattern if not mmproj_allow_pattern else f"{allow_pattern}__{mmproj_allow_pattern}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    return f"{safe}.downloaded.json"


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
    else:
        manifest_name = _manifest_name(allow_pattern, mmproj_allow_pattern)
        base = Path(output_dir)

        # Idempotency contract: the per-request manifest is the only marker
        # of completeness. If it verifies, skip. Otherwise wipe any
        # previously-matched files and the stale manifest before
        # re-downloading so a partial run never lingers on disk.
        if is_complete(str(base), manifest_name=manifest_name):
            emit_json_info(f"Model already downloaded: {repo_id} ({allow_pattern})")
            return
        stale_manifest = base / manifest_name
        if stale_manifest.is_file():
            stale_manifest.unlink()
        for f in _matched_files(base, allow_pattern):
            f.unlink()
        if mmproj_allow_pattern:
            for f in _matched_files(base, mmproj_allow_pattern):
                f.unlink()

        os.makedirs(output_dir, exist_ok=True)

        tqdm_class = JsonProgress

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

        # Remove download caches
        cache_path = Path(output_dir) / ".cache"
        if cache_path.is_dir():
            shutil.rmtree(cache_path)

        # Write the sidecar manifest listing every file produced by this
        # request, so the next invocation can verify the download was
        # actually complete (presence + expected size) and not just a
        # partial extraction.
        downloaded_files = _matched_files(base, allow_pattern)
        if mmproj_allow_pattern:
            downloaded_files += _matched_files(base, mmproj_allow_pattern)
        if not downloaded_files:
            emit_json_error(
                f"No files matched after download for repository {repo_id} "
                f"(pattern: {allow_pattern})"
            )
            raise SystemExit(1)
        write_manifest(
            str(base),
            [str(f) for f in downloaded_files],
            model_id=os.environ["model_id"],
            manifest_name=manifest_name,
        )

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))


if __name__ == "__main__":
    main()
