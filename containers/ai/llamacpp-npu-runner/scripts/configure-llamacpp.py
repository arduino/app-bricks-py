# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
import sys
from pathlib import Path

# A Hexagon session can map about 1.8 GiB of repacked weights, so the number of sessions a
# model needs is driven by how much of it lands on the NPU, not by its parameter count: the
# repacked size ranges from 44% to 103% of the GGUF depending on the architecture (token
# embeddings stay on the CPU, matformer models share a lot of weights). The parameter count
# is wrong in both directions, so size the sessions on the GGUF size instead.
#
# Number of sessions by GGUF size, ordered from the largest threshold down: the first entry a
# model exceeds wins. GB here means 10^9 bytes. The table never under-allocates on the models
# we ship (it over-allocates one session on a few of them, which costs ~3% per token); models
# with a known-good value should carry an explicit GGML_HEXAGON_NDEV instead.
NDEV_BY_GGUF_GB = ((5.0, 4), (3.5, 3), (1.5, 2))

# Models that must stay on a single session no matter their size: splitting across sessions
# happens layer by layer, and these have too few layers to fill more than one. Matched as a
# substring of the model name (the GGUF file stem), so a match covers every quantization.
SINGLE_SESSION_MODELS = ("gemma-4-E2B",)


def model_ndev(name: str, gguf_bytes: int) -> int:
    """Return the number of Hexagon sessions required by model name, sized gguf_bytes bytes."""
    if any(single in name for single in SINGLE_SESSION_MODELS):
        return 1

    gguf_gb = gguf_bytes / 1e9
    for threshold, ndev in NDEV_BY_GGUF_GB:
        if gguf_gb > threshold:
            return ndev
    return 1


def detect_hexagon_ndev(models):
    """Return the number of Hexagon sessions required by the installed models.

    Diagnostics go to stderr so that stdout can carry just the number.
    """
    ndev = 1
    for name, entry in sorted(models.items()):
        try:
            gguf_bytes = Path(entry["model"]).stat().st_size
        except OSError as e:
            print(f"  {name}: cannot read size ({e}), assuming it fits 1 session", file=sys.stderr)
            continue

        required = model_ndev(name, gguf_bytes)
        gguf_gb = gguf_bytes / 1e9
        if any(single in name for single in SINGLE_SESSION_MODELS):
            print(f"  {name}: {gguf_gb:.2f} GB, pinned to 1 session (too few layers to split)", file=sys.stderr)
        elif required == 1:
            print(f"  {name}: {gguf_gb:.2f} GB, fits 1 session", file=sys.stderr)
        else:
            print(f"  {name}: {gguf_gb:.2f} GB, requires {required} sessions", file=sys.stderr)
        ndev = max(ndev, required)

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
