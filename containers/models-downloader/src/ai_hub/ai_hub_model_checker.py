# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Verify an AI Hub model download against its manifest.

Exits 0 with an ``info`` event when the manifest exists and every file it
lists is present on disk with the expected size; exits 1 with an ``error``
event otherwise.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.http_download import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an AI Hub model download.")
    parser.add_argument(
        "--model-directory",
        default=os.environ.get("model_directory"),
        help="Model sub-directory under /models (defaults to the "
             "'model_directory' environment variable).",
    )
    parser.add_argument(
        "--models-root",
        default="/models",
        help="Root directory where models live (default: /models).",
    )
    args = parser.parse_args()

    if not args.model_directory:
        print(json.dumps({
            "event": "error",
            "description": "Model directory is not set",
        }), flush=True)
        sys.exit(1)

    directory = os.path.join(args.models_root, args.model_directory)
    if not os.path.isdir(directory):
        print(json.dumps({
            "event": "error",
            "description": f"Model does not exist: {args.model_directory}",
        }), flush=True)
        sys.exit(1)

    ok, reason = verify_manifest(directory)
    if not ok:
        print(json.dumps({
            "event": "error",
            "description": f"Model is incomplete: {args.model_directory} ({reason})",
        }), flush=True)
        sys.exit(1)

    print(json.dumps({
        "event": "info",
        "description": f"Model exists: {args.model_directory}",
    }), flush=True)


if __name__ == "__main__":
    main()
