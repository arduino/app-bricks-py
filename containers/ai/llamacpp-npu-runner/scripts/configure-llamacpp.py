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
# known-good value should carry an explicit GGML_HEXAGON_DEVICES instead.
#
# Before resizing these tables over a load failure, check the board's free RAM first: the DSP
# buffers come from plain system memory (/dev/dma_heap/system), so "ggml-hex: fastrpc_mmap
# failed" with error 0x1 on a buffer that used to fit usually means the board was out of RAM
# at that moment — other models resident in the router, leftover instances — not that the
# model needs more sessions. Measured on a 21q: a loaded model costs ~1.6x its GGUF in RAM.

# Default table, for the context sizes the service runs at out of the box.
NDEV_BY_GGUF_GB = ((3.5, 4), (1.5, 2))

# Small-context table. A 4k KV cache leaves far more room on the domains. It was
# originally measured on a ventunoq board (every installed model loaded at 1..4 sessions
# with -c 4096, taking the first count that loads):
#
#   Qwen3.5-0.8B-Q4_0    0.51 GB -> 1     Qwen3-8B-Q4_0        4.79 GB -> 2
#   Qwen3-4B-2507-Q4_0   2.38 GB -> 1     Qwen3.5-9B-Q4_0      5.74 GB -> 2
#   Qwen3.5-4B-Q4_0      2.78 GB -> 1     gemma-4-12b          6.98 GB -> 3
#
# Qwen3-8B-Q4_0 has since failed to fastrpc-map its per-domain buffer on 2 sessions
# (possibly RAM pressure rather than session sizing, see above — not re-measured yet),
# so pending a re-measurement of the multi-session rows on a quiet board, every model
# above the 1-session rows takes everything the hardware has.
SMALL_CTX_SIZE = 4096
NDEV_BY_GGUF_GB_SMALL_CTX = ((3.5, 4),)


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


# The per-download record the models-downloader writes next to what it fetches.
# It is what tells an out-of-the-box model from a downloaded one, so the served
# names need no catalog baked into this image.
METADATA_NAME = ".arduino_metadata.yaml"


def downloaded_records(directory: Path):
    """The download records of *directory*'s ".arduino_metadata.yaml", newest last.

    The document is ``models: [...]``, one record per model downloaded into the
    directory (see the models-downloader's common/model_metadata.py). A missing or
    unusable file yields no records — the models there are then out-of-the-box.

    Mirrors the models-downloader's ``metadata_records``, keep the two in sync. This
    code is duplicated in the llamacpp-runner and llamacpp-npu-runner images.
    """
    try:
        import yaml

        with open(directory / METADATA_NAME) as f:
            data = yaml.safe_load(f)
    except Exception:
        return []
    records = data.get("models") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def file_record(gguf_file: Path, models_dir: Path):
    """The download record describing *gguf_file*, or None when no record names it.

    The record lives in the directory the download landed in — for Hugging Face the
    repository directory, which can sit above a nested per-quantization folder — so
    every directory from the file's own up to *models_dir* is tried. Within a record,
    ``files`` holds paths relative to the record's directory; they are matched by
    full relative path or by basename, the two ways download patterns match a file.
    """
    directory = gguf_file.parent
    while True:
        rel = gguf_file.relative_to(directory).as_posix()
        for record in downloaded_records(directory):
            files = record.get("files")
            if isinstance(files, list) and any(isinstance(f, str) and (f == rel or f.split("/")[-1] == gguf_file.name) for f in files):
                return record
        if directory == models_dir or directory == directory.parent:
            return None
        directory = directory.parent


def gguf_model_name(gguf_file: Path, models_dir: Path) -> str:
    """The name llama-server serves this file under.

    Decided by the file's download record: a user-configured model (downloaded ad
    hoc) is named by its models_dir-relative path, so same-named files from different
    repositories never collide; a curated download (model_origin "built_in") keeps its
    file stem. A file with no record at all is an out-of-the-box model and keeps its
    stem too — that is the fallback, records only exist for downloaded models.

    The models-downloader derives the ``llamacpp:<name>`` ids of the same files, and
    the LLM brick resolves those against these sections, so the two namings may never
    drift apart. This code is duplicated in the llamacpp-runner and llamacpp-npu-runner
    images.
    """
    record = file_record(gguf_file, models_dir)
    if record is not None and record.get("model_origin") == "user":
        return gguf_file.relative_to(models_dir).with_suffix("").as_posix()
    return gguf_file.stem


def find_models(models_dir: Path):
    """Return {model name: {"model": path, "mmproj": path}} for every model in models_dir."""
    models = {}

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        entry = {"model": gguf_file.as_posix()}

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            entry["mmproj"] = mmproj_files[0].as_posix()

        models[gguf_model_name(gguf_file, models_dir)] = entry

    return models


def generate_models_ini(models_dir: Path):
    """Write the models.ini preset indexing every model in models_dir."""
    config = configparser.ConfigParser()
    config.read_dict(find_models(models_dir))

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
        print(detect_ctx_size(find_models(args.models_dir), args.ctx))
    elif args.print_ndev:
        scope = f"a {args.ctx} token context" if args.ctx > 0 else "the default context"
        print(f"Scanning installed models to size the Hexagon sessions for {scope}...", file=sys.stderr)
        print(detect_hexagon_ndev(find_models(args.models_dir), args.ctx))
    else:
        generate_models_ini(args.models_dir)


if __name__ == "__main__":
    main()
