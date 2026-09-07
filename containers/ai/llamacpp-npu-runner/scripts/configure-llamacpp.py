# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Configure llama.cpp for the models installed in a directory.

Writes the models.ini preset the server is started with, and answers the two questions
run-model-router.sh asks before starting it: which context size the installed models can
hold (``--print-ctx``), and how many Hexagon sessions they need at that context
(``--print-ndev``). Those two modes print a single number on stdout and send every
diagnostic to stderr, so the caller can read the answer with a command substitution.
``--probe-ggml-types`` is internal: the script runs itself in that mode to read the ggml
type table out of libggml in a child process (see probe_ggml_types()).

Usage:
    python configure-llamacpp.py /models
    python configure-llamacpp.py /models --print-ctx --ctx 16384
    python configure-llamacpp.py /models --print-ndev --ctx 4096
    python configure-llamacpp.py --probe-ggml-types /opt/pkg-snapdragon/lib/libggml-base.so
"""

import argparse
import configparser
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# --------------------------------------------------------------------------- #
# GGUF inspection
#
# Session sizing needs two numbers per model: how many bytes of weights the Hexagon
# backend puts on the NPU, and how big the KV cache is at the context the server runs
# at. Both are in the GGUF header, so they are read from there rather than guessed from
# the file size, which is off by up to a factor of three in either direction (an
# all-q4_0 model puts nearly every byte on the NPU, the matformer gemmas a third).
#
# Only the header and the tensor index are read, never the weights.
# --------------------------------------------------------------------------- #

GGUF_MAGIC = b"GGUF"

# (name, block size, bytes per block) per ggml type, indexed by the type id GGUF stores.
# The tensor index carries a type and a shape but no byte count, so a tensor's size has to
# be computed from these: elements / block size * bytes per block.
#
# This is only the fallback. The image ships the very ggml that will read these files, and
# ggml_types() below asks it, so that a llama.cpp bump which adds a quantization needs no
# edit here — this table was already three types out of date one bump after being written.
# fmt: off
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1),
    25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54),
    35: ("tq2_0", 256, 66), 39: ("mxfp4", 32, 17), 40: ("nvfp4", 64, 36),
    41: ("q1_0", 128, 18), 42: ("q2_0", 64, 18),
}
# fmt: on

# Type ids are probed up to here at most; the probe stops earlier, at the end of the enum
# (see probe_ggml_types_of()). ggml exports no count.
MAX_GGML_TYPE_ID = 64

# How long the probe may take before it is given up on, in seconds. It takes well under
# one; the limit is there so that a hung child cannot hold up the container start.
PROBE_TIMEOUT = 60

_ggml_types = None


def ggml_types() -> dict:
    """(name, block size, bytes per block) per ggml type id.

    Read from the libggml this image ships, whose answers are by definition the ones that
    apply to the files it will load, and from GGML_TYPES when that library cannot be
    loaded or answers implausibly. Probed once and remembered.
    """
    global _ggml_types
    if _ggml_types is None:
        probed = probe_ggml_types()
        _ggml_types = {**GGML_TYPES, **probed} if probed else dict(GGML_TYPES)
    return _ggml_types


def find_ggml_library() -> Path | None:
    """The libggml-base shared object on LD_LIBRARY_PATH, or None when there is none."""
    return next(
        (
            candidate
            for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if directory
            for candidate in sorted(Path(directory).glob("libggml-base.so*"))
        ),
        None,
    )


def probe_ggml_types() -> dict:
    """Ask libggml for its type table, or return {} if it cannot be asked.

    The asking happens in a child process (this script in --probe-ggml-types mode),
    because it is the one piece of native code on the sizing path and it has crashed:
    ggml_type_name() is not bounds-checked, so a type id past the end of the enum makes it
    return whatever pointer follows the type table, and dereferencing that segfaulted the
    runner on the 20260902 build (type ids 56 and 58). In-process, that took the sizing
    down with it and the server started on one session for an 8B model. The child prints
    one type per line as it goes, so whatever it read before dying still counts, and the
    built-in table covers the rest — nothing here is worth failing a container start over.
    """
    library = find_ggml_library()
    if library is None:
        return {}
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--probe-ggml-types", str(library)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  libggml probe could not run ({type(e).__name__}: {e}), sizing with the built-in type table", file=sys.stderr)
        return {}

    table = parse_probed_types(result.stdout)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["no output"]
        print(
            f"  libggml probe died with exit status {result.returncode} ({detail[0]}) after reading "
            f"{len(table)} type(s): the built-in type table covers the rest",
            file=sys.stderr,
        )
    return table


def parse_probed_types(output: str) -> dict:
    """The type table from the lines a --probe-ggml-types child printed, malformed ones skipped."""
    table = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        try:
            type_id, block, size = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            continue
        table[type_id] = (fields[1], block, size)
    return table


def implausible_ggml_type(name: str, block: int, size: int) -> bool:
    """Whether an answer looks like memory past the enum, or a gap in it, rather than a type."""
    return not (name and name.replace("_", "").isalnum() and 1 <= block <= 1024 and 1 <= size <= 4096)


def probe_ggml_types_of(ggml, known_types: dict = GGML_TYPES, limit: int = MAX_GGML_TYPE_ID):
    """Yield (id, name, block size, bytes per block) for every type *ggml* knows, in order.

    *ggml* answers ggml_type_name(), ggml_blck_size() and ggml_type_size() like the
    library does. The enum has gaps — removed types, which answer with a name and a zero
    block size — and no end marker: the first implausible answer past the last type this
    file knows is taken as the end, and nothing past it is read, since that is what
    crashes. A type added after a future gap is therefore not seen, which is safe: a
    tensor of an unknown type is refused rather than sized as free (see read_gguf()).
    """
    last_known = max(known_types)
    for type_id in range(limit):
        name = (ggml.ggml_type_name(type_id) or b"").decode("ascii", "replace")
        block, size = ggml.ggml_blck_size(type_id), ggml.ggml_type_size(type_id)
        if implausible_ggml_type(name, block, size):
            if type_id > last_known:
                return
            continue
        yield type_id, name, int(block), int(size)


def print_probed_ggml_types(library: Path) -> None:
    """--probe-ggml-types mode: load *library* and print one "id name block size" line per type."""
    import ctypes

    ggml = ctypes.CDLL(str(library))
    ggml.ggml_type_name.restype, ggml.ggml_type_name.argtypes = ctypes.c_char_p, [ctypes.c_int]
    ggml.ggml_blck_size.restype, ggml.ggml_blck_size.argtypes = ctypes.c_int64, [ctypes.c_int]
    ggml.ggml_type_size.restype, ggml.ggml_type_size.argtypes = ctypes.c_size_t, [ctypes.c_int]
    for type_id, name, block, size in probe_ggml_types_of(ggml):
        # One line per type, flushed at once, so that a crash on the next id loses nothing.
        print(f"{type_id}\t{name}\t{block}\t{size}", flush=True)


# Fixed-size metadata value types, by the type id GGUF stores: (struct format, size).
# fmt: off
GGUF_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
# fmt: on
GGUF_BOOL, GGUF_STRING, GGUF_ARRAY = 7, 8, 9

# Metadata arrays longer than this are token vocabularies: they are read past without
# being kept, so that a 150k-entry vocabulary costs no memory here.
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
            items = [self.value(item_type) for _ in range(count)]
            return items if count <= MAX_KEPT_ARRAY else None
        raise ValueError(f"unknown GGUF value type {value_type}")


class Tensor(NamedTuple):
    """One entry of the GGUF tensor index."""

    name: str
    type: str
    bytes: int


def read_gguf(path: Path):
    """Return (metadata, tensors) from a GGUF file's header, without reading weights."""
    with open(path, "rb") as f:
        reader = GgufReader(f)
        if reader.raw(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        reader.u32()  # header version
        tensor_count, metadata_count = reader.u64(), reader.u64()

        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.value(reader.u32())

        tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [reader.u64() for _ in range(reader.u32())]
            type_id = reader.u32()
            reader.u64()  # offset into the tensor data, which is never read
            types = ggml_types()
            if type_id not in types:
                # Sizing a tensor of an unknown type as zero bytes would under-count the
                # model and start the server with too few sessions to load it. Refusing
                # the header instead falls back to sizing by file size, see model_sizings().
                raise ValueError(f"unknown ggml tensor type {type_id} for tensor {name!r}")
            type_name, block, block_bytes = types[type_id]
            tensors.append(Tensor(name, type_name, math.prod(dimensions) // block * block_bytes))

    return metadata, tensors


# --------------------------------------------------------------------------- #
# Hexagon session sizing
#
# A model is spread over the Hexagon sessions layer by layer: each session holds the
# weights of its layers, the slice of the KV cache that belongs to them, and its own
# compute buffers. Sizing it is a matter of keeping the busiest session under what the DSP
# will reliably map for one, which is not a single number — see ../SESSION_ALLOCATION.md,
# where loads succeeded holding 2.9 GiB and failed holding 1.8 GiB. Past the limit the
# load fails with
#
#   ggml-hex: HTP0 buffer mapping failed : domain_id 3 size 536883200 error 0x00000001
#   alloc_tensor_range: failed to allocate HTP0 buffer of size 536870912
#
# Before resizing anything over such a failure, check the board's free RAM: the DSP
# buffers come from plain system memory (/dev/dma_heap/system), so the same message
# also appears when the board is simply out of it. Measured: a loaded model
# costs about 1.6x its GGUF size in RAM.
# --------------------------------------------------------------------------- #

MIB = 1024 * 1024

# Bytes of weights plus KV cache that one Hexagon session holds. Calibrated on the load
# measurements in ../SESSION_ALLOCATION.md, against the worst observation of each
# configuration rather than the luckiest: no configuration that ever failed to load asks
# for this little. The binding one is Nemotron-Mini-4B at 16k over two sessions, which
# wants 1838 MiB and failed, so the margin here is thin — but lowering it further starts
# capping contexts that were measured to work, which costs more than a spare session.
SESSION_BUDGET = 1800 * MIB

# HTP0..HTP3: the Hexagon sessions the cDSP firmware gives one process.
MAX_SESSIONS = 4

# Bytes an element of the KV cache costs, by the type llama-server is configured to store
# it in (LLAMA_ARG_CACHE_TYPE_K / _V, f16 unless the service sets them). The quantized
# types cost their block size: q8_0 is 34 bytes per 32 elements, 53% of what f16 costs,
# which takes 47% off the whole context axis. See ../SESSION_ALLOCATION.md.
# fmt: off
KV_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
    "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}
# fmt: on
DEFAULT_KV_TYPE = "f16"
"""What llama.cpp stores the cache in when nothing says otherwise, so what to size for."""


def kv_type(which: str) -> str:
    """The type the K or V cache will be stored in: "K" or "V"."""
    return os.environ.get(f"LLAMA_ARG_CACHE_TYPE_{which}", DEFAULT_KV_TYPE).strip().lower()


def kv_cache_description() -> str:
    """The cache types, for the diagnostics: "f16", or "q8_0/f16" when the two differ."""
    k, v = kv_type("K"), kv_type("V")
    return k if k == v else f"{k}/{v}"


def kv_element_bytes(which: str) -> float:
    """Bytes per element of the K or V cache.

    A type this table does not know — one llama.cpp has gained since, not a typo, since
    llama-server validates the value and refuses to start on an unknown one — is sized
    as f32, the largest of them: sizing a cache smaller than it turns out to be is what
    fails a load, while sizing it larger only costs a session.
    """
    return KV_TYPE_BYTES.get(kv_type(which), KV_TYPE_BYTES["f32"])


# The shape llama-server runs at, which decides how many cells a sliding-window KV cache
# holds (see KvLayer.cells()). Both are read from the environment the server will see, with
# llama-server's own defaults behind them: LLAMA_ARG_N_PARALLEL unset means the slot count
# is auto-sized, which comes to 4, and service_compose.yaml sets LLAMA_ARG_UBATCH.
DEFAULT_N_SEQ_MAX = 4
DEFAULT_N_UBATCH = 256


def env_int(name: str, default: int) -> int:
    """A positive integer from the environment, or *default* when unset or not one."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def n_seq_max() -> int:
    return env_int("LLAMA_ARG_N_PARALLEL", DEFAULT_N_SEQ_MAX)


def n_ubatch() -> int:
    return env_int("LLAMA_ARG_UBATCH", DEFAULT_N_UBATCH)


# The Hexagon backend holds every quantization but the K-quants, which it does not repack:
# those stay on the CPU.
K_QUANT_SUFFIX = "_K"

# A hybrid model (Qwen3.5, LFM2) keeps a recurrent state per sequence on the NPU next to
# the KV cache of its attention layers. Its size follows the architecture rather than the
# context, and measured between 1 and 201 MiB, so it is covered by a flat allowance instead
# of being derived from the ssm.* metadata.
RECURRENT_STATE_ALLOWANCE = 256 * MIB


class KvLayer(NamedTuple):
    """One layer's KV cache: what it costs per cell, and over how many cells."""

    bytes_per_cell: int

    window: int
    """Sliding attention window in tokens, or 0 when the layer attends to everything."""

    def cells(self, ctx_size: int) -> int:
        """Cells llama.cpp allocates for this layer at a given context size.

        A windowed layer only keeps its window, plus room for the sequences in flight;
        that is what lets the gemmas hold a 16k context in a KV cache sized for 2k.
        """
        if self.window <= 0:
            return ctx_size
        return min(ctx_size, n_seq_max() * self.window + n_ubatch())

    def bytes(self, ctx_size: int) -> int:
        return self.bytes_per_cell * self.cells(ctx_size)


class ModelShape(NamedTuple):
    """What session sizing needs to know about a model, read from its GGUF header."""

    n_layer: int
    npu_weight_bytes: int
    kv_layers: tuple[KvLayer, ...]
    state_bytes: int

    def kv_cache_bytes(self, ctx_size: int) -> int:
        return sum(layer.bytes(ctx_size) for layer in self.kv_layers)

    def npu_bytes(self, ctx_size: int) -> int:
        """Everything the model puts on the NPU at a given context size."""
        return self.npu_weight_bytes + self.kv_cache_bytes(ctx_size) + self.state_bytes

    def session_bytes(self, ctx_size: int, sessions: int) -> int:
        """Bytes the busiest of *sessions* sessions ends up holding.

        Weights and KV cache are split by layer, and the busiest session takes one
        layer's worth more than an even share — which is what the per-session figures
        llama-server logs show: granite-4.2-3b over two sessions puts 21 of its 40
        layers' worth of KV cache on HTP0, measured as 672 MiB of 1280 at 16k.
        """
        layers_per_session = -(-self.n_layer // sessions) + 1
        share = min(1.0, layers_per_session / self.n_layer)
        return int(self.npu_bytes(ctx_size) * share)

    def sessions_needed(self, ctx_size: int) -> int:
        """Hexagon sessions this model needs at *ctx_size*, MAX_SESSIONS at worst."""
        for count in range(1, MAX_SESSIONS):
            if self.session_bytes(ctx_size, count) <= SESSION_BUDGET:
                return count
        return MAX_SESSIONS

    def fits(self, ctx_size: int) -> bool:
        """Whether the model fits the budget on any number of sessions at all."""
        return self.session_bytes(ctx_size, MAX_SESSIONS) <= SESSION_BUDGET

    def describe(self, ctx_size: int) -> str:
        """How this model was sized, for the diagnostics."""
        per_session = self.session_bytes(ctx_size, self.sessions_needed(ctx_size)) / MIB
        return (
            f"{self.npu_weight_bytes / MIB:.0f} MiB on the NPU"
            f" + {self.kv_cache_bytes(ctx_size) / MIB:.0f} MiB of {kv_cache_description()} KV cache"
            f" -> {per_session:.0f} MiB per session" + ("" if self.fits(ctx_size) else f", over budget even on {sessions(MAX_SESSIONS)}")
        )


# --------------------------------------------------------------------------- #
# Sizing by file size, for a model whose header cannot be read
#
# These are the tables the runner used before it read GGUF headers, kept as a fallback so
# that an unreadable or unrecognised file degrades to the previous behaviour rather than
# to no sizing at all. They are keyed on GGUF size in GB (10^9 bytes), ordered from the
# largest threshold down, and deliberately over-allocate rather than risk a failed load:
# the file size is a poor proxy for what reaches the NPU, which is the whole reason the
# header is read when it can be.
# --------------------------------------------------------------------------- #

GB = 1e9
FALLBACK_SESSIONS_BY_GGUF_GB = ((3.5, 4), (1.5, 3))
FALLBACK_SESSIONS_BY_GGUF_GB_SMALL_CTX = ((5.0, 4), (3.5, 3))
FALLBACK_SMALL_CTX_SIZE = 4096

# The models the tables get wrong and the sessions they were pinned to, matched as a
# substring of the model name so that every quantization is covered. The matformer gemmas
# put a third of their bytes on the NPU, so sizing them by file over-allocates; at these
# counts they also never trigger the context cap, which is what the tables would do.
FALLBACK_PINS = (("gemma-4-E2B", 1), ("gemma-4-E4B", 2))


class FileSizeSizing(NamedTuple):
    """Sizing for a model that could only be measured by its name and the size of its file."""

    name: str
    gguf_bytes: int

    def sessions_needed(self, ctx_size: int) -> int:
        for pinned, count in FALLBACK_PINS:
            if pinned in self.name:
                return count
        table = FALLBACK_SESSIONS_BY_GGUF_GB_SMALL_CTX if 0 < ctx_size <= FALLBACK_SMALL_CTX_SIZE else FALLBACK_SESSIONS_BY_GGUF_GB
        for threshold, count in table:
            if self.gguf_bytes / GB > threshold:
                return count
        return 1

    def describe(self, ctx_size: int) -> str:
        pinned = any(name in self.name for name, _ in FALLBACK_PINS)
        return f"{self.gguf_bytes / GB:.2f} GB on disk, {'pinned by name' if pinned else 'sized by file size'}"


def npu_weight_bytes(tensors) -> int:
    """Bytes of weights the Hexagon backend holds on the NPU.

    Everything but the K-quants, and but the token embeddings of a model that has a
    separate output projection: those are only read by get_rows and stay on the CPU. A
    model with tied embeddings uses that same tensor as its output projection, and it
    does go to the NPU. Reproduces the "HTP model buffer size" llama-server logs to
    within a MiB on 18 of the 19 models measured.
    """
    names = {tensor.name for tensor in tensors}
    dropped = set()
    if {"output.weight", "token_embd.weight"} <= names:
        dropped.add("token_embd.weight")
    return sum(tensor.bytes for tensor in tensors if not tensor.type.endswith(K_QUANT_SUFFIX) and tensor.name not in dropped)


K_PROJECTION = re.compile(r"^blk\.(\d+)\.attn_k(?:_b)?\.weight$")
FUSED_QKV_PROJECTION = re.compile(r"^blk\.(\d+)\.attn_qkv\.weight$")


def kv_layer_indices(metadata, tensors) -> list[int]:
    """Indices of the layers that own a KV cache, from the tensor index.

    A layer owns one when the index has its own K projection: the recurrent layers of a
    hybrid model do not, and neither do the layers of a gemma that share another layer's
    cache. Models that fuse Q, K and V into one tensor (phi3) are read from that
    instead, but only when no layer has a separate K projection at all — Qwen3.5 gives
    its recurrent layers a fused projection too, and those cache nothing.

    Falls back to every layer when nothing matches, which is the safe direction: an
    unrecognised naming scheme then sizes as if every layer cached.
    """
    for pattern in (K_PROJECTION, FUSED_QKV_PROJECTION):
        indices = sorted({int(m.group(1)) for tensor in tensors if (m := pattern.match(tensor.name))})
        if indices:
            return indices
    return list(range(block_count(metadata)))


def block_count(metadata) -> int:
    """The model's layer count, 0 when the header does not say."""
    count = metadata.get(f"{metadata.get('general.architecture', '')}.block_count", 0)
    return int(count) if isinstance(count, (int, float)) else 0


def kv_layers(metadata, tensors) -> tuple[KvLayer, ...]:
    """The KV cache of every layer that owns one.

    A layer counts as windowed only where the GGUF says so layer by layer: a model that
    declares a window without the per-layer pattern is sized as if it attended to
    everything, which over-estimates it. Reproduces every "llama_kv_cache: size =" line
    measured, exactly for the models that carry the pattern.
    """
    architecture = metadata.get("general.architecture", "")

    def key(name, default=None):
        return metadata.get(f"{architecture}.{name}", default)

    def at(value, layer, default=0):
        """A metadata value that is either one number or one number per layer."""
        if isinstance(value, list):
            return value[layer] if layer < len(value) else default
        return value if isinstance(value, (int, float)) else default

    pattern = key("attention.sliding_window_pattern")
    window = key("attention.sliding_window") or 0
    heads, embedding = key("attention.head_count"), key("embedding_length")

    layers = []
    for layer in kv_layer_indices(metadata, tensors):
        windowed = isinstance(pattern, list) and bool(at(pattern, layer))
        suffix = "_swa" if windowed else ""
        k = int(at(key(f"attention.key_length{suffix}"), layer))
        v = int(at(key(f"attention.value_length{suffix}"), layer)) or k
        if not k and heads and embedding:
            # No explicit head size: the heads divide the embedding, as in llama.cpp.
            k = v = int(at(embedding, layer)) // max(1, int(at(heads, layer, 1)))
        kv_heads = max(1, int(at(key("attention.head_count_kv"), layer, 1)))
        cell = kv_heads * (k * kv_element_bytes("K") + v * kv_element_bytes("V"))
        layers.append(KvLayer(int(cell), window if windowed else 0))

    return tuple(layers)


RECURRENT_TENSOR = re.compile(r"\.(?:ssm_|shortconv|conv1d)")


def read_model_shape(path: Path) -> ModelShape:
    """Read what session sizing needs to know about the model in *path*.

    Raises when the header parses but says nothing a model would say — a well-formed
    header with no tensors and no layer count is a truncated or empty file, and sizing it
    from what it claims would come to no weights at all. The caller falls back to sizing
    by file size, which cannot be fooled this way.
    """
    metadata, tensors = read_gguf(path)
    if not tensors or not block_count(metadata) or not any(t.name.startswith("blk.") for t in tensors):
        raise ValueError(f"no model in the header: {len(tensors)} tensors, {block_count(metadata)} layers")
    layers = kv_layers(metadata, tensors)
    # A layer that caches nothing is either recurrent, which costs a state, or sharing
    # another layer's cache, which costs nothing: the tensor names tell the two apart.
    recurrent = any(RECURRENT_TENSOR.search(tensor.name) for tensor in tensors)
    return ModelShape(
        n_layer=max(block_count(metadata), len(layers), 1),
        npu_weight_bytes=npu_weight_bytes(tensors),
        kv_layers=layers,
        state_bytes=RECURRENT_STATE_ALLOWANCE if recurrent else 0,
    )


# --------------------------------------------------------------------------- #
# Context sizing
#
# The context is a server-wide setting, and the KV cache of a large one is what pushes
# the big models past the sessions available: a model that needs every session at the
# requested context gets a smaller one instead. The fourth session is the cutoff
# because that is where the DSP allocator was measured to stop being reproducible —
# two of the five 16k configurations that need four sessions failed to load, one of
# them for a model whose byte-identical twin loaded (see ../SESSION_ALLOCATION.md).
# --------------------------------------------------------------------------- #

# Sessions a model may need before the context is cut down for it.
CTX_CAP_MIN_SESSIONS = MAX_SESSIONS

# The context is halved down to this floor, never below it.
MIN_CTX_SIZE = 4096

# Context the server runs at when none is configured (--ctx 0): the one the service
# configures out of the box (service_compose.yaml). run-model-router.sh exports it as
# LLAMA_ARG_CTX_SIZE too, rather than leaving llama-server to its own default, so that the
# sessions are sized for the context the server really gets.
DEFAULT_CTX_SIZE = 16384


def sessions(count: int) -> str:
    """Return a session count as it reads in the diagnostics ("1 session", "2 sessions")."""
    return f"{count} session" if count == 1 else f"{count} sessions"


def model_sizings(models):
    """Yield (name, sizing) for every installed model, header-derived where possible.

    A model whose header cannot be read, or reads implausibly, falls back to being sized
    by file size: less accurate, but it is what the runner did before it read headers, and
    it always produces a number. Skipping the model instead would be the one unsafe
    outcome, since the server would then be started with too few sessions to load it.
    """
    for name, entry in sorted(models.items()):
        path = Path(entry["model"])
        try:
            yield name, read_model_shape(path)
            continue
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        try:
            fallback = FileSizeSizing(name, path.stat().st_size)
        except OSError as e:
            print(f"  {name}: cannot read it at all ({e}), ignoring it", file=sys.stderr)
            continue
        print(f"  {name}: cannot size it from its GGUF header ({reason}), falling back to its file size", file=sys.stderr)
        yield name, fallback


def detect_hexagon_sessions(models, ctx_size: int) -> int:
    """Return the number of Hexagon sessions the installed models need at *ctx_size*."""
    needed = 1
    for name, sizing in model_sizings(models):
        required = sizing.sessions_needed(ctx_size)
        verdict = "fits 1 session" if required == 1 else f"requires {sessions(required)}"
        print(f"  {name}: {sizing.describe(ctx_size)}, {verdict}", file=sys.stderr)
        needed = max(needed, required)

    return needed


def detect_ctx_size(models, ctx_size: int) -> int:
    """Return the context size the server should run at.

    The requested one, halved for as long as a model needs CTX_CAP_MIN_SESSIONS to hold
    it, down to MIN_CTX_SIZE. A context of 0 means "not configured": llama-server picks
    it, so there is nothing to cap.
    """
    if ctx_size <= MIN_CTX_SIZE:
        return ctx_size

    sizings = list(model_sizings(models))
    effective = ctx_size
    while effective > MIN_CTX_SIZE:
        hungry = [name for name, sizing in sizings if sizing.sessions_needed(effective) >= CTX_CAP_MIN_SESSIONS]
        if not hungry:
            break
        for name in hungry:
            print(f"  {name}: needs {sessions(CTX_CAP_MIN_SESSIONS)} to hold {effective} tokens", file=sys.stderr)
        effective = max(MIN_CTX_SIZE, effective // 2)
        print(f"  capping the context to {effective}", file=sys.stderr)

    if effective == ctx_size:
        print(f"  every model holds the {ctx_size} token context", file=sys.stderr)
    return effective


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
    parser.add_argument("models_dir", type=Path, nargs="?", help="Path to the models directory")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--probe-ggml-types",
        type=Path,
        metavar="LIBGGML",
        help="Internal: print the type table of the given libggml shared object, one "
        "'id name block size' line per type, instead of generating models.ini",
    )
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
        f"leaves for the weights on each session (0: not configured, taken as {DEFAULT_CTX_SIZE})",
    )
    return parser.parse_args()


def main():
    """Run the mode selected on the command line."""
    args = parse_args()

    if args.probe_ggml_types is not None:
        print_probed_ggml_types(args.probe_ggml_types)
        return

    if args.models_dir is None or not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    ctx_size = args.ctx if args.ctx > 0 else DEFAULT_CTX_SIZE
    scope = f"a {ctx_size} token context" if args.ctx > 0 else f"an unconfigured context (taken as {ctx_size})"
    if args.print_ctx:
        print(f"Scanning installed models to check they hold {scope}...", file=sys.stderr)
        print(detect_ctx_size(find_models(args.models_dir), ctx_size))
    elif args.print_ndev:
        print(f"Scanning installed models to size the Hexagon sessions for {scope}...", file=sys.stderr)
        print(detect_hexagon_sessions(find_models(args.models_dir), ctx_size))
    else:
        generate_models_ini(args.models_dir)


if __name__ == "__main__":
    main()
