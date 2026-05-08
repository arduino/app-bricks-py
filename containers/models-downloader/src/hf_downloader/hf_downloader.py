# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os

from huggingface_hub import snapshot_download
import argparse
import configparser
from pathlib import Path
from tqdm.auto import tqdm
import json


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

    print(f"Generated {output_path} with {len(config.sections())} model(s)")


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
        "--json-progress",
        action="store_true",
        help="Report progress as JSON lines, e.g. {'progress': '42%%'}, instead of a progress bar.",
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
            print(f"Repository ID: {repo_id}")
            print(f"Model key: {args.model_key}")
            print(f"Model type: {model_type}")
            print(f"Quantization: {quantization}")
            if mmproj_quantization:
                print(f"MMProj Quantization: {mmproj_quantization[0]}")

        allow_pattern = f"*{quantization}*.gguf"
        mmproj_allow_pattern = f"*mmproj*{mmproj_quantization[0]}*.gguf" if mmproj_quantization else None
    else:
        if not args.model_repo_id or not args.model_name:
            raise ValueError("If --model-key is not provided, both --model-repo-id and --model-name must be specified")

        repo_id = args.model_repo_id

        allow_pattern = args.model_name
        if allow_pattern == "":
            raise ValueError("model name cannot be empty")
        if not allow_pattern.contains("*") and not allow_pattern.endswith(".gguf"):
            allow_pattern = f"*{allow_pattern}*"

        if args.model_mmproj_name and args.model_mmproj_name != "":
            mmproj_allow_pattern = args.model_mmproj_name
            if not mmproj_allow_pattern.contains("*") and not mmproj_allow_pattern.endswith(".gguf"):
                mmproj_allow_pattern = f"*{mmproj_allow_pattern}*"

        if args.verbose:
            print(f"Repository ID: {repo_id}")
            print(f"Model identifier: {allow_pattern}")
            if mmproj_allow_pattern:
                print(f"MMProj file: {mmproj_allow_pattern}")

    if args.hf_token and args.hf_token != "":
        os.environ["HF_HUB_TOKEN"] = args.hf_token

    # Create download folder if it doesn't exist. Patter is: output_dir + / repo_id
    output_dir = f"{args.output_dir}/{repo_id}"
    os.makedirs(output_dir, exist_ok=True)

    tqdm_class = JsonProgress
    if not args.json_progress:
        tqdm_class = None

    # Download the model using Hugging Face API
    if args.verbose:
        print(f"Downloading model from Hugging Face repository: {repo_id} with allow pattern: {allow_pattern}")
    snapshot_download(repo_id=repo_id, allow_patterns=[allow_pattern], ignore_patterns=["*mmproj*"], local_dir=output_dir, tqdm_class=tqdm_class)

    if mmproj_allow_pattern:
        if args.verbose:
            print(f"Downloading mmproj model file from Hugging Face repository: {repo_id} with allow pattern: {mmproj_allow_pattern}")
        snapshot_download(repo_id=repo_id, allow_patterns=[mmproj_allow_pattern], local_dir=output_dir, tqdm_class=tqdm_class)

    # Generate models.ini file
    generate_models_ini(Path(args.output_dir))


if __name__ == "__main__":
    main()
