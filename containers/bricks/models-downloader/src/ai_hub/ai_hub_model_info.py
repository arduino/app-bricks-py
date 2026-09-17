# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Return model_size_mb from models-list.yaml for a given ai-hub-handler model.

Looks up the model by matching model_type and model_name in the deployment variables.
Prints a JSON stat event with size_mb if found, or size_mb -1 if not found.

Usage:
    python ai_hub_model_info.py --model-type genie --model-name qwen3_4b_instruct_2507
    python ai_hub_model_info.py --model-type voice_ai --model-name whisper_small_quantized --model-list /app/models-list.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models_list import load_models_list, find_model_size_mb, MODELS_LIST_PATH


def main():
    parser = argparse.ArgumentParser(description="Return model_size_mb from models-list.yaml for an ai-hub-handler model.")
    parser.add_argument(
        "--model-type",
        required=True,
        type=str,
        metavar="TYPE",
        help="AI Hub model type (e.g. genie, voice_ai).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        metavar="NAME",
        help="AI Hub model name (e.g. qwen3_4b_instruct_2507).",
    )
    parser.add_argument(
        "--model-list",
        default=MODELS_LIST_PATH,
        metavar="PATH",
        help=f"Path to models-list.yaml (default: {MODELS_LIST_PATH}).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model_list):
        print(
            json.dumps({
                "event": "error",
                "description": f"models-list.yaml not found at {args.model_list}",
            }),
            flush=True,
        )
        sys.exit(1)

    model_key = f"{args.model_type}:{args.model_name}"

    try:
        models = load_models_list(args.model_list)
    except Exception as exc:
        print(
            json.dumps({
                "event": "error",
                "description": f"Failed to load models-list.yaml: {exc}",
            }),
            flush=True,
        )
        sys.exit(1)

    size_mb = find_model_size_mb(models, args.model_type, args.model_name)

    print(
        json.dumps({
            "event": "stat",
            "description": f"Model info for {model_key}",
            "size_mb": size_mb,
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
