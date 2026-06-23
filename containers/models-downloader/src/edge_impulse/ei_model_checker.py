# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Verify an Edge Impulse model download against its sidecar manifest.

Exits 0 with an ``info`` event when the manifest exists and the model file
is present on disk with the expected size; exits 1 with an ``error`` event
otherwise.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an Edge Impulse model download.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("model_name"),
        help="Model filename under /models (defaults to the 'model_name' "
             "environment variable).",
    )
    parser.add_argument(
        "--models-root",
        default="/models",
        help="Root directory where models live (default: /models).",
    )
    args = parser.parse_args()

    if not args.model_name:
        print(json.dumps({
            "event": "error",
            "description": "Model name is not set",
        }), flush=True)
        sys.exit(1)

    model_path = os.path.join(args.models_root, args.model_name)
    if not os.path.isfile(model_path):
        print(json.dumps({
            "event": "error",
            "description": f"Model does not exist: {args.model_name}",
        }), flush=True)
        sys.exit(1)

    ok, reason = verify_manifest(
        args.models_root,
        manifest_name=f"{args.model_name}.downloaded.json",
    )
    if not ok:
        print(json.dumps({
            "event": "error",
            "description": f"Model is incomplete: {args.model_name} ({reason})",
        }), flush=True)
        sys.exit(1)

    print(json.dumps({
        "event": "info",
        "description": f"Model exists: {args.model_name}",
    }), flush=True)


if __name__ == "__main__":
    main()
