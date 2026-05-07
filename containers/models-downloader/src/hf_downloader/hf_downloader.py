# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os

from huggingface_hub import snapshot_download
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download an Hugging Face model via HF download API")
    parser.add_argument(
        "--model-key",
        required=True,
        type=str,
        metavar="KEY",
        help="model key (e.g. llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16). "
        "The format is: <model_type>:<repo_id>:<quantization>:<optional mmproj quantization>.",
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

    args = parser.parse_args()
    model_type, repo_id, quantization, *mmproj_quantization = args.model_key.split(":")
    if repo_id == "":
        raise ValueError("repo_id cannot be empty")
    if quantization == "":
        raise ValueError("quantization cannot be empty")

    print(f"Starting download for model: {args.model_key}")

    if args.verbose:
        print(f"Downloading model: {args.model_key}")
        print(f"Model type: {model_type}")
        print(f"Repository ID: {repo_id}")
        print(f"Quantization: {quantization}")
        if mmproj_quantization:
            print(f"MMProj Quantization: {mmproj_quantization[0]}")

    if args.hf_token and args.hf_token != "":
        os.environ["HF_HUB_TOKEN"] = args.hf_token

    # Create download folder if it doesn't exist. Patter is: output_dir + / repo_id
    output_dir = f"{args.output_dir}/{repo_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Download the model using Hugging Face API
    print(f"Downloading model from Hugging Face repository: {repo_id} with quantization: {quantization}")
    snapshot_download(repo_id=repo_id, allow_patterns=f"*{quantization}*.gguf", ignore_patterns=["*mmproj*"], local_dir=output_dir)

    if mmproj_quantization and len(mmproj_quantization) != 0 and mmproj_quantization[0] != "":
        snapshot_download(repo_id=repo_id, allow_patterns=f"*mmproj*{mmproj_quantization[0]}*.gguf", local_dir=output_dir)


if __name__ == "__main__":
    main()
