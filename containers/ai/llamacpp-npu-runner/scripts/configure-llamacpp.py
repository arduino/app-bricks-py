# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Configure llama.cpp for the models installed in a directory.

Writes the models.ini preset the server is started with, and answers the two questions
run-model-router.sh asks before starting it: which context size the installed models can
hold (``--print-ctx``), and how many Hexagon sessions they need at that context
(``--print-ndev``). Those two modes print a single number on stdout and send every
diagnostic to stderr, so the caller can read the answer with a command substitution.

Usage:
    python configure-llamacpp.py /models
    python configure-llamacpp.py /models --print-ctx --ctx 16384
    python configure-llamacpp.py /models --print-ndev --ctx 4096
"""

import argparse
import configparser
import sys
from pathlib import Path
from typing import NamedTuple

# The tables below are written in GB, meaning 10^9 bytes.
GB = 1e9


# --------------------------------------------------------------------------- #
# Hexagon session sizing
# --------------------------------------------------------------------------- #

# A Hexagon session maps about 1 GiB of repacked weights per DSP domain, so the number of
# sessions a model needs is driven by how much of it lands on the NPU, not by its parameter
# count: the repacked size ranges from roughly 44% to 103% of the GGUF depending on the
# architecture (token embeddings stay on the CPU, matformer models share a lot of weights).
# The parameter count is wrong in both directions, so size the sessions on the GGUF size.
#
# The KV cache is allocated on the same domains as the weights, so the sizing also depends
# on the context size: hence two tables, picked by ndev_table() below.
#
# Number of sessions by GGUF size, ordered from the largest threshold down: the first entry a
# model exceeds wins. Both tables are deliberately conservative — they over-allocate a session
# on some models, which costs ~3% per token, rather than failing to load; models with a
# known-good value should carry an explicit GGML_HEXAGON_NDEV instead.

# Default table, for the context sizes the service runs at out of the box.
NDEV_BY_GGUF_GB = ((3.5, 4), (1.5, 2))

# Small-context table. A 4k KV cache leaves far more room on the domains, and this one is
# measured rather than estimated: on a ventunoq board every installed model was loaded at
# 1..4 sessions with -c 4096, taking the first count that loads.
#
#   Qwen3.5-0.8B-Q4_0    0.51 GB -> 1     Qwen3-8B-Q4_0        4.79 GB -> 2
#   Qwen3-4B-2507-Q4_0   2.38 GB -> 1     Qwen3.5-9B-Q4_0      5.74 GB -> 2
#   Qwen3.5-4B-Q4_0      2.78 GB -> 1     gemma-4-12b          6.98 GB -> 3
#
# Thresholds sit between the measurements, so the table reproduces all of them exactly.
# Above 8 GB there is no measurement, so that bucket gets everything the hardware has.
SMALL_CTX_SIZE = 4096
NDEV_BY_GGUF_GB_SMALL_CTX = ((8.0, 4), (6.0, 3), (3.5, 2))


class ModelOverride(NamedTuple):
    """A model the tables above get wrong, and what to do with it instead."""

    name: str
    """Matched as a substring of the model name (the GGUF file stem), so a match covers
    every quantization of the model. The first override that matches wins."""

    ndev: int
    """Number of Hexagon sessions to pin the model to, whatever its GGUF size."""

    holds_full_ctx: bool
    """Whether the model keeps the full context: it never triggers the cap below."""


# The matformer gemmas share most of their weights and repack to far less than their GGUF
# size, so sizing them by file size over-allocates, while splitting happens layer by layer,
# so they cannot spread over an arbitrary number of sessions either: E2B holds on a single
# session, E4B needs two, and both keep the full context on those sessions.
MODEL_OVERRIDES = (
    ModelOverride("gemma-4-E2B", ndev=1, holds_full_ctx=True),
    ModelOverride("gemma-4-E4B", ndev=2, holds_full_ctx=True),
)


def model_override(name: str) -> ModelOverride | None:
    """Return the override matching this model name, or None if the tables apply to it."""
    return next((override for override in MODEL_OVERRIDES if override.name in name), None)


def ndev_table(ctx_size: int):
    """Return the (GGUF size, sessions) table to use for a context size.

    A context of 0 means "not configured", which falls back to the default table.
    """
    if 0 < ctx_size <= SMALL_CTX_SIZE:
        return NDEV_BY_GGUF_GB_SMALL_CTX
    return NDEV_BY_GGUF_GB


def model_ndev(name: str, gguf_bytes: int, ctx_size: int) -> int:
    """Return the number of Hexagon sessions required by model name, sized gguf_bytes bytes."""
    override = model_override(name)
    if override:
        return override.ndev

    for threshold, ndev in ndev_table(ctx_size):
        if gguf_bytes / GB > threshold:
            return ndev
    return 1


# --------------------------------------------------------------------------- #
# Context sizing
# --------------------------------------------------------------------------- #

# A model that needs this many sessions has so little room left on the domains that the KV
# cache of a large context does not fit: installing one caps the context the server runs at
# down to the size the small-context table above is measured at.
CTX_CAP_MIN_NDEV = 4
CAPPED_CTX_SIZE = SMALL_CTX_SIZE


def sessions(ndev: int) -> str:
    """Return a session count as it reads in the diagnostics ("1 session", "2 sessions")."""
    return f"{ndev} session" if ndev == 1 else f"{ndev} sessions"


def model_sizes(models):
    """Yield (name, GGUF size in bytes) for every installed model whose file can be read."""
    for name, entry in sorted(models.items()):
        try:
            yield name, Path(entry["model"]).stat().st_size
        except OSError as e:
            print(f"  {name}: cannot read size ({e}), ignoring it", file=sys.stderr)


def detect_ctx_size(models, ctx_size: int) -> int:
    """Return the context size the server should run at, capped if a big model is installed.

    A context of 0 means "not configured": llama-server picks it, so there is nothing to cap.
    The models flagged holds_full_ctx never cause the cap. Note that the context is a
    server-wide setting, so that exempts them from *causing* it; a big model without the flag
    installed alongside them still caps the context for everyone.
    """
    if ctx_size <= CAPPED_CTX_SIZE:
        return ctx_size

    for name, gguf_bytes in model_sizes(models):
        gguf_gb = gguf_bytes / GB
        override = model_override(name)
        if override and override.holds_full_ctx:
            print(f"  {name}: {gguf_gb:.2f} GB, exempt from the context cap", file=sys.stderr)
            continue

        required = model_ndev(name, gguf_bytes, ctx_size)
        if required >= CTX_CAP_MIN_NDEV:
            capped = f"capping the context to {CAPPED_CTX_SIZE}"
            print(f"  {name}: {gguf_gb:.2f} GB, needs {sessions(required)} at {ctx_size} tokens: {capped}", file=sys.stderr)
            return CAPPED_CTX_SIZE

    print(f"  no model needs {sessions(CTX_CAP_MIN_NDEV)}: keeping the {ctx_size} token context", file=sys.stderr)
    return ctx_size


def detect_hexagon_ndev(models, ctx_size: int) -> int:
    """Return the number of Hexagon sessions required by the installed models."""
    ndev = 1
    for name, gguf_bytes in model_sizes(models):
        required = model_ndev(name, gguf_bytes, ctx_size)
        if model_override(name):
            verdict = f"pinned to {sessions(required)}"
        elif required == 1:
            verdict = "fits 1 session"
        else:
            verdict = f"requires {sessions(required)}"
        print(f"  {name}: {gguf_bytes / GB:.2f} GB, {verdict}", file=sys.stderr)
        ndev = max(ndev, required)

    return ndev


# --------------------------------------------------------------------------- #
# models.ini generation
# --------------------------------------------------------------------------- #


# The curated catalog, baked into the image, decides the served model names
MODELS_LIST_PATH = "/models-list.yaml"


def curated_declarations(models_list_path):
    """The (model_directory, filename) locations that models-list.yaml declares for llamacpp.
    A GGUF at a declared location is a curated model and keeps its file stem as the served
    name; every other file is ad-hoc and is named by its path.

    ``filename`` is None when the entry's model_url pins no file (a compact key naming
    a quantization); any GGUF inside the directory then counts as declared.

    An unreadable catalog degrades to no declarations: every model is then named by its path,
    and curated models fail instead of being served under an ambiguous stem.

    Mirrors the models-downloader's common/gguf_naming.py, keep the two in sync. This code
    is duplicated in the llamacpp-runner and llamacpp-npu-runner images.
    """
    try:
        import yaml

        with open(models_list_path) as f:
            entries = (yaml.safe_load(f) or {}).get("models", [])
    except Exception:
        return []

    declarations = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        for model_data in entry.values():
            if not isinstance(model_data, dict):
                continue
            deployment = model_data.get("deployment")
            platforms = deployment.get("platforms") if isinstance(deployment, dict) else None
            for platform_entry in platforms or []:
                if not isinstance(platform_entry, dict):
                    continue
                for platform_config in platform_entry.values():
                    variables = platform_config.get("variables") if isinstance(platform_config, dict) else None
                    if not isinstance(variables, dict):
                        continue
                    repository = str(variables.get("models_repository") or "")
                    if repository.rsplit("/models/", 1)[-1].removeprefix("models/") != "llamacpp":
                        continue
                    directory = str(variables.get("model_directory") or "").strip("/")
                    if not directory:
                        continue
                    url = str(variables.get("model_url") or "")
                    base = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
                    declaration = (directory, base if base.endswith(".gguf") else None)
                    if declaration not in declarations:
                        declarations.append(declaration)
    return declarations


def gguf_model_name(gguf_file: Path, models_dir: Path, declarations) -> str:
    """The name llama-server serves this file under (see curated_declarations)."""
    rel = gguf_file.relative_to(models_dir)
    rel_dir = rel.parent.as_posix()
    rel_dir = "" if rel_dir == "." else rel_dir
    for directory, filename in declarations:
        if rel_dir != directory and not rel_dir.startswith(directory + "/"):
            continue
        if filename is None or filename == gguf_file.name:
            return gguf_file.stem
    return rel.with_suffix("").as_posix()


def find_models(models_dir: Path, models_list_path: str = MODELS_LIST_PATH):
    """Return {model name: {"model": path, "mmproj": path}} for every model in models_dir."""
    models = {}
    declarations = curated_declarations(models_list_path)

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        entry = {"model": gguf_file.as_posix()}

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            entry["mmproj"] = mmproj_files[0].as_posix()

        models[gguf_model_name(gguf_file, models_dir, declarations)] = entry

    return models


def generate_models_ini(models_dir: Path, models_list_path: str = MODELS_LIST_PATH):
    """Write the models.ini preset indexing every model in models_dir."""
    config = configparser.ConfigParser()
    config.read_dict(find_models(models_dir, models_list_path))

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    print(f"Generated {output_path} with {len(config.sections())} model(s)")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Generate models.ini from a models directory")
    parser.add_argument("models_dir", type=Path, help="Path to the models directory")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--print-ndev",
        action="store_true",
        help="Print only the number of Hexagon sessions required by the installed models "
        "on stdout (diagnostics go to stderr) instead of generating models.ini",
    )
    mode.add_argument(
        "--print-ctx",
        action="store_true",
        help="Print only the context size the server should run at on stdout: the one passed "
        "with --ctx, capped for the big models that cannot hold it (diagnostics go to stderr)",
    )

    parser.add_argument(
        "--models-list",
        default=MODELS_LIST_PATH,
        help=f"Path to the curated models-list.yaml (default: {MODELS_LIST_PATH})",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="Context size the server will run at, which decides how much room the KV cache "
        f"leaves for the weights on each session (0: unknown, sizes as if above {SMALL_CTX_SIZE})",
    )
    return parser.parse_args()


def main():
    """Run the mode selected on the command line."""
    args = parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    if args.print_ctx:
        print(f"Scanning installed models to check they hold a {args.ctx} token context...", file=sys.stderr)
        print(detect_ctx_size(find_models(args.models_dir, args.models_list), args.ctx))
    elif args.print_ndev:
        scope = f"a {args.ctx} token context" if args.ctx > 0 else "the default context"
        print(f"Scanning installed models to size the Hexagon sessions for {scope}...", file=sys.stderr)
        print(detect_hexagon_ndev(find_models(args.models_dir, args.models_list), args.ctx))
    else:
        generate_models_ini(args.models_dir, args.models_list)


if __name__ == "__main__":
    main()
