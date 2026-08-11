# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
import sys
from pathlib import Path

# A Hexagon session maps about 1 GiB of repacked weights per DSP domain, so the number of
# sessions a model needs is driven by how much of it lands on the NPU, not by its parameter
# count: the repacked size ranges from roughly 44% to 103% of the GGUF depending on the
# architecture (token embeddings stay on the CPU, matformer models share a lot of weights).
# The parameter count is wrong in both directions, so size the sessions on the GGUF size.
#
# The KV cache is allocated on the same domains as the weights, so the sizing also depends
# on the context size: hence two profiles, picked by ndev_profile() below.
#
# Number of sessions by GGUF size, ordered from the largest threshold down: the first entry a
# model exceeds wins. GB here means 10^9 bytes. Both tables are deliberately conservative —
# they over-allocate a session on some models, which costs ~3% per token, rather than failing
# to load; models with a known-good value should carry an explicit GGML_HEXAGON_NDEV instead.

# Default profile, for the context sizes the service runs at out of the box.
NDEV_BY_GGUF_GB = ((3.5, 4), (1.5, 2))

# Models that must stay on a single session no matter their size: splitting across sessions
# happens layer by layer, and these have too few layers to fill more than one. Matched as a
# substring of the model name (the GGUF file stem), so a match covers every quantization.
SINGLE_SESSION_MODELS = ("gemma-4-E2B",)

# Small-context profile. A 4k KV cache leaves far more room on the domains, and this table is
# measured rather than estimated: on a ventunoq board every installed model was loaded at
# 1..4 sessions with -c 4096, taking the first count that loads.
#
#   Qwen3.5-0.8B-Q4_0    0.51 GB -> 1     Qwen3-8B-Q4_0        4.79 GB -> 2
#   Qwen3-4B-2507-Q4_0   2.38 GB -> 1     gemma-4-E4B          5.15 GB -> 1
#   Qwen3.5-4B-Q4_0      2.78 GB -> 1     Qwen3.5-9B-Q4_0      5.74 GB -> 2
#   gemma-4-E2B          3.35 GB -> 1     gemma-4-12b          6.98 GB -> 3
#
# Thresholds sit between the measurements, so the table reproduces all of them exactly.
# Above 8 GB there is no measurement, so that bucket gets everything the hardware has.
SMALL_CTX_SIZE = 4096
NDEV_BY_GGUF_GB_SMALL_CTX = ((8.0, 4), (6.0, 3), (3.5, 2))

# The matformer gemmas repack to far less than their file size, measured at 4k: E4B fits a
# single session despite being the second largest GGUF we ship.
SINGLE_SESSION_MODELS_SMALL_CTX = ("gemma-4-E2B", "gemma-4-E4B")


def ndev_profile(ctx_size: int):
    """Return the (size table, single-session models) pair to use for a context size.

    A context of 0 means "not configured", which falls back to the default profile.
    """
    if 0 < ctx_size <= SMALL_CTX_SIZE:
        return NDEV_BY_GGUF_GB_SMALL_CTX, SINGLE_SESSION_MODELS_SMALL_CTX
    return NDEV_BY_GGUF_GB, SINGLE_SESSION_MODELS


def model_ndev(name: str, gguf_bytes: int, ctx_size: int = 0) -> int:
    """Return the number of Hexagon sessions required by model name, sized gguf_bytes bytes."""
    table, single_session = ndev_profile(ctx_size)
    if any(single in name for single in single_session):
        return 1

    gguf_gb = gguf_bytes / 1e9
    for threshold, ndev in table:
        if gguf_gb > threshold:
            return ndev
    return 1


def detect_hexagon_ndev(models, ctx_size: int = 0):
    """Return the number of Hexagon sessions required by the installed models.

    Diagnostics go to stderr so that stdout can carry just the number.
    """
    _, single_session = ndev_profile(ctx_size)
    ndev = 1
    for name, entry in sorted(models.items()):
        try:
            gguf_bytes = Path(entry["model"]).stat().st_size
        except OSError as e:
            print(f"  {name}: cannot read size ({e}), assuming it fits 1 session", file=sys.stderr)
            continue

        required = model_ndev(name, gguf_bytes, ctx_size)
        gguf_gb = gguf_bytes / 1e9
        if any(single in name for single in single_session):
            print(f"  {name}: {gguf_gb:.2f} GB, pinned to 1 session", file=sys.stderr)
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
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="Context size the server will run at, which decides how much room the KV cache "
        f"leaves for the weights on each session (0: unknown, sizes as if above {SMALL_CTX_SIZE})",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    if args.print_ndev:
        scope = f"a {args.ctx} token context" if args.ctx > 0 else "the default context"
        print(f"Scanning installed models to size the Hexagon sessions for {scope}...", file=sys.stderr)
        print(detect_hexagon_ndev(find_models(args.models_dir), args.ctx))
    else:
        generate_models_ini(args.models_dir)
