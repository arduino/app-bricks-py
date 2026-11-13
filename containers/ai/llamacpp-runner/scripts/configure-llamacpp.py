# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Configure llama.cpp for the models installed in a directory.

Writes the models.ini preset the server is started with (--models-preset, router mode):
one section per GGUF, named as the LLM brick addresses it (see gguf_model_name()), with
the file as ``model``, an mmproj companion from the same directory as ``mmproj``, and, for
a decision model, the LLAMA_ARG_* keys that give its prompt a whole micro-batch of its
own (see decision_model_options()). Diagnostics go to stderr.

Usage:
    python configure-llamacpp.py /models
"""

import argparse
import configparser
import os
import struct
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# GGUF inspection
#
# Only the metadata of the header is read, never the tensor index or the weights: enough
# to tell a decision model from a chat model. The llamacpp-npu-runner's copy of this reader
# goes on to read the tensor index for its Hexagon session sizing.
# --------------------------------------------------------------------------- #

GGUF_MAGIC = b"GGUF"

# Fixed-size metadata value types, by the type id GGUF stores: (struct format, size).
# fmt: off
GGUF_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
# fmt: on
GGUF_BOOL, GGUF_STRING, GGUF_ARRAY = 7, 8, 9
GGUF_STRING_LENGTH = struct.Struct("<Q")

# Metadata arrays longer than this are token vocabularies: they are skipped without being
# decoded, since a header holds up to ~1.3M of those values (token and merge strings,
# scores, types) and every installed model is read at start.
MAX_KEPT_ARRAY = 1024


class GgufReader:
    """Sequential reader for the GGUF header encoding."""

    def __init__(self, file):
        self.file = file

    def raw(self, count: int) -> bytes:
        data = self.file.read(count)
        if len(data) != count:
            raise ValueError("truncated GGUF header")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, value_type: int):
        """One metadata value."""
        if value_type in GGUF_SCALARS:
            fmt, size = GGUF_SCALARS[value_type]
            number = struct.unpack(fmt, self.raw(size))[0]
            return bool(number) if value_type == GGUF_BOOL else number
        if value_type == GGUF_STRING:
            return self.string()
        if value_type == GGUF_ARRAY:
            item_type, count = self.u32(), self.u64()
            if count > MAX_KEPT_ARRAY:
                self.skip(item_type, count)
                return None
            return [self.value(item_type) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {value_type}")

    def skip(self, item_type: int, count: int) -> None:
        """Move past *count* values of *item_type* without decoding them."""
        if item_type in GGUF_SCALARS:
            self.file.seek(count * GGUF_SCALARS[item_type][1], os.SEEK_CUR)
        elif item_type == GGUF_STRING:
            # Once per token and merge: the calls are bound locally rather than made
            # through u64().
            read, seek, length = self.file.read, self.file.seek, GGUF_STRING_LENGTH.unpack
            for _ in range(count):
                seek(length(read(8))[0], os.SEEK_CUR)
        else:
            for _ in range(count):
                self.value(item_type)


def read_gguf_header(reader: GgufReader) -> tuple[int, dict]:
    """Read the fixed part of a GGUF header: magic, version, counts and metadata.

    Leaves *reader* at the tensor index and returns (tensor count, metadata).
    """
    if reader.raw(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    reader.u32()  # header version
    tensor_count, metadata_count = reader.u64(), reader.u64()

    metadata = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = reader.value(reader.u32())
    return tensor_count, metadata


def read_gguf_metadata(path: Path) -> dict:
    """The metadata key/values of a GGUF file's header, without its tensor index or weights.

    Enough to tell what kind of model the file holds (see decision_model_options()). The
    llamacpp-npu-runner's session sizing reads the tensor index too, with its read_gguf().
    This code is duplicated in the llamacpp-runner and llamacpp-npu-runner images.
    """
    with open(path, "rb") as f:
        return read_gguf_header(GgufReader(f))[1]


# Context the server runs at when none is configured: the one the service configures out
# of the box (service_compose.yaml, LLAMA_ARG_CTX_SIZE), so what to size a preset for.
DEFAULT_CTX_SIZE = 16384


def env_int(name: str, default: int) -> int:
    """A positive integer from the environment, or *default* when unset or not one."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# --------------------------------------------------------------------------- #
# Decision models
#
# A decision model (llama.cpp's /v1/systemone, marked by the <arch>.decision.type key of
# its header) does not generate text: it answers with probabilities. Its header, not its
# architecture, says whether it is causal: d3-edge, d3-lite and d3-nano are Qwen3.5 models with
# <arch>.attention.causal false, as a ModernBERT has, while Kev is a Qwen3.5 without the
# key. A non-causal one evaluates its whole prompt in one micro-batch, since every token
# attends to every other: a prompt longer than the server's micro-batch fails outright,
# and the service configures a small one for the chat models (LLAMA_ARG_UBATCH=128 on the
# UNO Q). So a non-causal decision model gets a preset of its own in models.ini, a batch
# as big as its prompt room: d3-edge, d3-lite and d3-nano are the models this preset serves today.
# A causal one (Kev) reads its prompt in chunks like a chat model and keeps the router's
# arguments: measured on the UNO Q, a 2048-token micro-batch only pushed it past the 2500m
# limit of the service (OOM-killed on a 4000-token state) where the chat models' 128
# serves it fine.
# --------------------------------------------------------------------------- #

# Tokens of prompt a non-causal decision model gets room for: its batch and its micro-batch,
# since it evaluates the whole prompt in one. Capped by the model's own context (a ModernBERT
# holds 8192). Measured on the UNO Q with a 0.42B ModernBERT: 523 MiB of RSS at this size,
# and a 3453-token state is refused by the server with a clear error rather than crashing.
DECISION_MODEL_BATCH = 2048


def decision_model_options(gguf_file: Path) -> dict[str, str]:
    """Per-model preset keys for a non-causal decision model, {} for anything else or an unreadable header.

    The keys are LLAMA_ARG_* environment variable names, which is how a llama-server
    --models-preset spells the arguments of one model (LLAMA_ARG_UBATCH renders to
    --ubatch-size for that child): they override the router's own for that model only.
    The context is the configured one — LLAMA_ARG_CTX_SIZE, or DEFAULT_CTX_SIZE when it
    is unset — never more than the model holds; the batch and the micro-batch are one
    and the same number, the prompt room, capped at DECISION_MODEL_BATCH. A causal
    decision model is told apart by its header (no <arch>.attention.causal false) and
    keeps the router's arguments.
    """
    try:
        metadata = read_gguf_metadata(gguf_file)
    except Exception:
        return {}
    arch = metadata.get("general.architecture")
    if not arch or f"{arch}.decision.type" not in metadata:
        return {}
    if metadata.get(f"{arch}.attention.causal") is not False:
        print(f"  {gguf_file.stem}: causal decision model ({metadata[f'{arch}.decision.type']}), served like a chat model", file=sys.stderr)
        return {}

    ctx = env_int("LLAMA_ARG_CTX_SIZE", DEFAULT_CTX_SIZE)
    model_ctx = metadata.get(f"{arch}.context_length")
    if isinstance(model_ctx, (int, float)) and model_ctx > 0:
        ctx = min(ctx, int(model_ctx))
    batch = min(ctx, DECISION_MODEL_BATCH)
    print(f"  {gguf_file.stem}: decision model ({metadata[f'{arch}.decision.type']}), ctx {ctx}, batch {batch}", file=sys.stderr)
    return {"LLAMA_ARG_CTX_SIZE": str(ctx), "LLAMA_ARG_BATCH": str(batch), "LLAMA_ARG_UBATCH": str(batch)}


# --------------------------------------------------------------------------- #
# models.ini generation
# --------------------------------------------------------------------------- #


# The per-download record the models-downloader writes next to what it fetches.
# It is what tells an out-of-the-box model from a downloaded one, so the served
# names need no catalog baked into this image.
METADATA_NAME = ".arduino_metadata.yaml"

# The name of a GGUF loaded with the multimodal projector next to it: the text model's
# name plus this suffix (the models-downloader's common/gguf_naming.py). This code is
# duplicated in the llamacpp-runner and llamacpp-npu-runner images.
MULTIMODAL_SUFFIX = "-multimodal"


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


def generate_models_ini(models_dir: Path):
    """Write the models.ini preset indexing every model in models_dir.

    A GGUF with a projector in its directory gets two sections: ``<name>``, the text
    model alone, and ``<name>-multimodal``, the same GGUF with the projector. A
    decision model's sections also carry its LLAMA_ARG_* keys, see decision_model_options().
    """
    config = configparser.ConfigParser()
    # The LLAMA_ARG_* keys of a decision model's section are environment variable names,
    # which llama-server matches case-sensitively; configparser lowercases keys otherwise.
    config.optionxform = str

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        section = gguf_model_name(gguf_file, models_dir)
        options = decision_model_options(gguf_file)
        config[section] = {"model": str(gguf_file.as_posix()), **options}

        # A projector in the same directory makes the multimodal model of this GGUF
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            config[section + MULTIMODAL_SUFFIX] = {"model": str(gguf_file.as_posix()), "mmproj": str(mmproj_files[0].as_posix()), **options}

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    print(f"Generated {output_path} with {len(config.sections())} model(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate models.ini from a models directory")
    parser.add_argument("models_dir", type=Path, help="Path to the models directory")
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    generate_models_ini(args.models_dir)
