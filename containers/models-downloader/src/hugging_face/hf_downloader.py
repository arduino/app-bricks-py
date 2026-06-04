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

Usage — two modes
-----------------
1. Compact key::

       hf_downloader --model-key llamacpp:<repo_id>:<quantization>[:<mmproj_quantization>]

2. Explicit names::

       hf_downloader --model-repo-id <repo_id> --model-name <file> [--model-mmproj-name <file>]

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
import shutil

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile
import argparse
import configparser
from pathlib import Path
from tqdm.auto import tqdm
import json


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
        if gguf_file.name.startswith("mmproj"):
            continue

        section = gguf_file.stem
        config[section] = {}
        config[section]["model"] = str(gguf_file.as_posix())

        # Look for mmproj file in the same directory
        mmproj_files = list(gguf_file.parent.glob("mmproj*.gguf"))
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
    if args.model_key and args.model_key != "":
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
        matched = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, allow_pattern)] if base.exists() else []
        if mmproj_allow_pattern:
            matched += [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, mmproj_allow_pattern)] if base.exists() else []
        if matched:
            emit_json_info(f"Model exists: {allow_pattern}")
        else:
            emit_json_error(f"Model does not exist: {allow_pattern}")
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
        os.makedirs(output_dir, exist_ok=True)

        tqdm_class = JsonProgress

        # Download the model using Hugging Face API
        if args.verbose:
            emit_json_info(f"Downloading model from Hugging Face repository: {repo_id} with allow pattern: {allow_pattern}")
        snapshot_download(repo_id=repo_id, allow_patterns=[allow_pattern], ignore_patterns=["*mmproj*"], local_dir=output_dir, tqdm_class=tqdm_class)

        if mmproj_allow_pattern:
            if args.verbose:
                emit_json_info(f"Downloading mmproj model file from Hugging Face repository: {repo_id} with allow pattern: {mmproj_allow_pattern}")
            snapshot_download(repo_id=repo_id, allow_patterns=[mmproj_allow_pattern], local_dir=output_dir, tqdm_class=tqdm_class)

        # Remove download caches
        cache_path = Path(output_dir) / ".cache"
        if cache_path.is_dir():
            shutil.rmtree(cache_path)

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))


if __name__ == "__main__":
    main()
