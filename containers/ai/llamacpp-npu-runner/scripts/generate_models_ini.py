# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
import re
import sys
from pathlib import Path

# Parameter count as written in the model name: "4B", "0.8B", "E4B" -> 4.0, 0.8, 4.0.
# The lookahead skips version numbers ("gemma-4-", "Qwen3.5") and quantization tags ("Q4_0");
# \b cannot be used because "E4B_q4_0" has a word character right after the "B".
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z0-9.])", re.IGNORECASE)

# Models with at least this many billion parameters need 2 Hexagon sessions to run on the NPU.
BIG_MODEL_PARAMS_B = 4.0


def model_params_b(model_name: str):
    """Return the parameter count (in billions) read from a model name, or None if unknown."""
    matches = _PARAMS_RE.findall(model_name)
    if not matches:
        return None
    return max(float(m) for m in matches)


def detect_hexagon_ndev(model_names):
    """Return the number of Hexagon sessions required by the installed models.

    Diagnostics go to stderr so that stdout can carry just the number.
    """
    ndev = 1
    for name in sorted(model_names):
        params = model_params_b(name)
        if params is None:
            print(f"  {name}: unknown parameter count, assuming it fits 1 session", file=sys.stderr)
        elif params >= BIG_MODEL_PARAMS_B:
            print(f"  {name}: {params}B parameters, requires 2 sessions", file=sys.stderr)
            ndev = 2
        else:
            print(f"  {name}: {params}B parameters, fits 1 session", file=sys.stderr)

    return ndev


def find_models(models_dir: Path):
    """Return {model name: {"model": path, "mmproj": path}} for every model in models_dir."""
    models = {}

    for gguf_file in sorted(models_dir.rglob("*.gguf")):
        if "mmproj" in gguf_file.name:
            continue

        entry = {"model": gguf_file.as_posix()}

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            entry["mmproj"] = mmproj_files[0].as_posix()

        models[gguf_file.stem] = entry

    return models


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()
    config.read_dict(find_models(models_dir))

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    print(f"Generated {output_path} with {len(config.sections())} model(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate models.ini from a models directory")
    parser.add_argument("models_dir", type=Path, help="Path to the models directory")
    parser.add_argument(
        "--print-ndev",
        action="store_true",
        help="Print only the number of Hexagon sessions required by the installed models "
        "on stdout (diagnostics go to stderr) instead of generating models.ini",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    if args.print_ndev:
        print("Scanning installed models to size the Hexagon sessions...", file=sys.stderr)
        print(detect_hexagon_ndev(find_models(args.models_dir)))
    else:
        generate_models_ini(args.models_dir)
