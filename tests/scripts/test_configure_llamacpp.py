# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the llama.cpp runners' configure-llamacpp.py scripts.

Covers the Hexagon session sizing of the NPU runner — the GGUF header reading it is
built on, the sizing itself, and a regression table of what the measured models were
measured to need (containers/ai/llamacpp-npu-runner/SESSION_ALLOCATION.md) — and, for
both runners, whose scripts deliberately duplicate the code, the served model names,
which are derived from the ".arduino_metadata.yaml" download records rather than from
a catalog baked into the images.
"""

from __future__ import annotations

import configparser
import importlib.util
import struct
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NPU_SCRIPT = REPO_ROOT / "containers" / "ai" / "llamacpp-npu-runner" / "scripts" / "configure-llamacpp.py"
CPU_SCRIPT = REPO_ROOT / "containers" / "ai" / "llamacpp-runner" / "scripts" / "configure-llamacpp.py"


def _load_script(script: Path, name: str):
    # The scripts are not importable by name (they live outside a package and have a dash).
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


configure_llamacpp = _load_script(NPU_SCRIPT, "configure_llamacpp")
configure_llamacpp_cpu = _load_script(CPU_SCRIPT, "configure_llamacpp_cpu")

KvLayer = configure_llamacpp.KvLayer
ModelShape = configure_llamacpp.ModelShape
read_model_shape = configure_llamacpp.read_model_shape
detect_hexagon_sessions = configure_llamacpp.detect_hexagon_sessions
detect_ctx_size = configure_llamacpp.detect_ctx_size

MIB = configure_llamacpp.MIB
CTX_SIZES = (4096, 8192, 16384)


# --------------------------------------------------------------------------- #
# GGUF headers, written the way llama.cpp writes them
#
# The sizing reads real headers, so the tests write real headers: a fake one would
# only prove that the fake matches the reader.
# --------------------------------------------------------------------------- #

GGUF_UINT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY = 4, 7, 8, 9
Q4_0, Q6_K, Q8_0, F32 = 2, 14, 8, 0


def _gguf_string(text: str) -> bytes:
    encoded = text.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value) -> bytes:
    """A metadata value, tagged with its GGUF type."""
    if isinstance(value, bool):
        return struct.pack("<IB", GGUF_BOOL, int(value))
    if isinstance(value, int):
        return struct.pack("<II", GGUF_UINT32, value)
    if isinstance(value, str):
        return struct.pack("<I", GGUF_STRING) + _gguf_string(value)
    if isinstance(value, list):
        item_type = GGUF_BOOL if isinstance(value[0], bool) else GGUF_UINT32
        body = b"".join(_gguf_value(item)[4:] for item in value)
        return struct.pack("<IIQ", GGUF_ARRAY, item_type, len(value)) + body
    raise TypeError(value)


def write_gguf(path: Path, metadata: dict, tensors: list[tuple[str, tuple[int, ...], int]]) -> Path:
    """Write a GGUF file holding only a header: metadata, and a tensor index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        out += _gguf_string(key) + _gguf_value(value)
    for name, dimensions, type_id in tensors:
        out += _gguf_string(name)
        out += struct.pack("<I", len(dimensions))
        out += b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
        out += struct.pack("<IQ", type_id, 0)
    path.write_bytes(bytes(out))
    return path


def q4_0_elements(mib: float) -> int:
    """Elements a q4_0 tensor needs to weigh *mib* MiB (18 bytes per 32 elements)."""
    return int(mib * MIB / 18) * 32


def attention_model(
    path: Path,
    *,
    layers: int,
    kv_heads: int = 8,
    head_size: int = 128,
    weight_mib: float = 100,
    recurrent_layers: int = 0,
    window: int = 0,
    architecture: str = "llama",
) -> Path:
    """A plain attention model: *layers* layers, all of them caching, plus the
    recurrent layers that do not. Its weights are one q4_0 tensor of *weight_mib*."""
    metadata = {
        "general.architecture": architecture,
        f"{architecture}.block_count": layers + recurrent_layers,
        f"{architecture}.attention.head_count": kv_heads,
        f"{architecture}.attention.head_count_kv": kv_heads,
        f"{architecture}.attention.key_length": head_size,
        f"{architecture}.attention.value_length": head_size,
        f"{architecture}.embedding_length": kv_heads * head_size,
    }
    if window:
        metadata[f"{architecture}.attention.sliding_window"] = window
        metadata[f"{architecture}.attention.sliding_window_pattern"] = [True] * layers
    tensors = [(f"blk.{layer}.attn_k.weight", (kv_heads * head_size,), Q4_0) for layer in range(layers)]
    tensors += [(f"blk.{layers + layer}.ssm_conv1d.weight", (16,), F32) for layer in range(recurrent_layers)]
    tensors.append(("weights.blob", (q4_0_elements(weight_mib),), Q4_0))
    return write_gguf(path, metadata, tensors)


# --------------------------------------------------------------------------- #
# What lands on the NPU
# --------------------------------------------------------------------------- #


def test_the_npu_holds_every_quantization_but_the_k_quants(tmp_path):
    """K-quants stay on the CPU: the Hexagon backend does not repack them."""
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {"general.architecture": "llama", "llama.block_count": 1},
        [
            ("blk.0.attn_k.weight", (q4_0_elements(10),), Q4_0),
            ("blk.0.ffn_up.weight", (256 * 1024,), Q6_K),
            ("blk.0.ffn_down.weight", (32 * 1024,), Q8_0),
        ],
    )

    shape = read_model_shape(gguf)

    q8_0_bytes = 32 * 1024 // 32 * 34
    assert shape.npu_weight_bytes == pytest.approx(10 * MIB + q8_0_bytes, rel=1e-3)


def test_the_token_embeddings_stay_on_the_cpu_when_the_output_is_a_separate_tensor(tmp_path):
    """Untied embeddings are only read by get_rows, which runs on the CPU."""
    tensors = [
        ("token_embd.weight", (q4_0_elements(300),), Q4_0),
        ("output.weight", (q4_0_elements(300),), Q4_0),
        ("blk.0.attn_k.weight", (q4_0_elements(10),), Q4_0),
    ]
    gguf = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "llama.block_count": 1}, tensors)

    assert read_model_shape(gguf).npu_weight_bytes == pytest.approx(310 * MIB, rel=1e-3)


def test_tied_token_embeddings_go_to_the_npu(tmp_path):
    """With no separate output.weight the same tensor is the output projection, and
    that one is a matmul the NPU does run."""
    tensors = [
        ("token_embd.weight", (q4_0_elements(300),), Q4_0),
        ("blk.0.attn_k.weight", (q4_0_elements(10),), Q4_0),
    ]
    gguf = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "llama.block_count": 1}, tensors)

    assert read_model_shape(gguf).npu_weight_bytes == pytest.approx(310 * MIB, rel=1e-3)


# --------------------------------------------------------------------------- #
# The KV cache
# --------------------------------------------------------------------------- #


def test_the_kv_cache_grows_with_the_context(tmp_path):
    """32 layers of 8 KV heads of 128, as f16: 128 KiB a token, 512 MiB at 4k."""
    gguf = attention_model(tmp_path / "m.gguf", layers=32, kv_heads=8, head_size=128)

    shape = read_model_shape(gguf)

    assert shape.kv_cache_bytes(4096) == 512 * MIB
    assert shape.kv_cache_bytes(8192) == 1024 * MIB
    assert shape.kv_cache_bytes(16384) == 2048 * MIB


def test_the_head_size_falls_back_to_the_embedding_split_over_the_heads(tmp_path):
    """granite states no key_length; llama.cpp divides the embedding by the heads."""
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "granite",
            "granite.block_count": 40,
            "granite.attention.head_count": 40,
            "granite.attention.head_count_kv": 8,
            "granite.embedding_length": 2560,
        },
        [(f"blk.{layer}.attn_k.weight", (512,), Q4_0) for layer in range(40)],
    )

    # 40 layers * 8 heads * (64 + 64) * 2 bytes = 80 KiB a token
    assert read_model_shape(gguf).kv_cache_bytes(8192) == 640 * MIB


def test_a_windowed_layer_stops_growing_at_its_window(tmp_path):
    """A sliding-window layer holds its window plus the sequences in flight, which is
    what lets the gemmas keep a 16k context in a cache sized for 2k."""
    gguf = attention_model(tmp_path / "m.gguf", layers=8, kv_heads=1, head_size=256, window=512)

    shape = read_model_shape(gguf)
    cells = configure_llamacpp.DEFAULT_N_SEQ_MAX * 512 + configure_llamacpp.DEFAULT_N_UBATCH

    assert shape.kv_cache_bytes(16384) == 8 * 1 * 512 * 2 * cells
    assert shape.kv_cache_bytes(16384) == shape.kv_cache_bytes(8192)
    # Below the window's own size the context is what binds.
    assert shape.kv_cache_bytes(1024) == 8 * 1 * 512 * 2 * 1024


def test_a_window_without_a_per_layer_pattern_is_sized_as_full_attention(tmp_path):
    """The safe direction: phi3 declares a 128k "window" that never binds, and gemma3
    states no pattern, so both are sized as if every layer attended to everything."""
    gguf = attention_model(tmp_path / "m.gguf", layers=8, kv_heads=1, head_size=256, window=512)
    del_pattern = read_model_shape(gguf).kv_cache_bytes(16384)

    metadata_only_window = write_gguf(
        tmp_path / "n.gguf",
        {
            "general.architecture": "gemma3",
            "gemma3.block_count": 8,
            "gemma3.attention.head_count": 1,
            "gemma3.attention.head_count_kv": 1,
            "gemma3.attention.key_length": 256,
            "gemma3.attention.value_length": 256,
            "gemma3.attention.sliding_window": 512,
        },
        [(f"blk.{layer}.attn_k.weight", (256,), Q4_0) for layer in range(8)],
    )

    full = read_model_shape(metadata_only_window).kv_cache_bytes(16384)
    assert full == 8 * 1 * 512 * 2 * 16384
    assert full > del_pattern


def test_only_the_layers_with_their_own_k_projection_cache(tmp_path):
    """Qwen3.5 gives every layer a fused QKV projection but only its attention layers
    a separate K projection; the recurrent ones cache nothing (measured: 8 of 32)."""
    tensors = [(f"blk.{layer}.attn_qkv.weight", (4096,), Q4_0) for layer in range(32)]
    tensors += [(f"blk.{layer}.attn_k.weight", (2048,), Q4_0) for layer in range(0, 32, 4)]
    tensors += [(f"blk.{layer}.ssm_conv1d.weight", (16,), F32) for layer in range(32)]
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 32,
            "qwen35.attention.head_count": 16,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
        },
        tensors,
    )

    shape = read_model_shape(gguf)

    assert len(shape.kv_layers) == 8
    # 8 layers * 4 heads * (256 + 256) * 2 bytes = 32 KiB a token
    assert shape.kv_cache_bytes(4096) == 128 * MIB
    assert shape.state_bytes == configure_llamacpp.RECURRENT_STATE_ALLOWANCE


def test_a_fused_qkv_projection_is_read_when_no_layer_has_a_separate_k(tmp_path):
    """phi3 fuses Q, K and V into one tensor in every layer, and every layer caches."""
    tensors = [(f"blk.{layer}.attn_qkv.weight", (4096,), Q4_0) for layer in range(32)]
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "phi3",
            "phi3.block_count": 32,
            "phi3.attention.head_count": 24,
            "phi3.attention.head_count_kv": 8,
            "phi3.attention.key_length": 128,
            "phi3.attention.value_length": 128,
        },
        tensors,
    )

    shape = read_model_shape(gguf)

    assert len(shape.kv_layers) == 32
    assert shape.kv_cache_bytes(8192) == 1024 * MIB
    assert shape.state_bytes == 0


def test_layers_that_share_a_cache_cost_no_recurrent_state(tmp_path):
    """The matformer gemmas leave most layers without a K projection because those
    share another layer's cache, not because they are recurrent."""
    tensors = [(f"blk.{layer}.attn_k.weight", (512,), Q4_0) for layer in range(15)]
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "gemma4",
            "gemma4.block_count": 35,
            "gemma4.attention.head_count": 8,
            "gemma4.attention.head_count_kv": 1,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
        },
        tensors,
    )

    shape = read_model_shape(gguf)

    assert len(shape.kv_layers) == 15
    assert shape.state_bytes == 0


def test_an_unreadable_header_is_not_a_model(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"not a gguf file")

    with pytest.raises(ValueError):
        read_model_shape(tmp_path / "m.gguf")


# --------------------------------------------------------------------------- #
# What happens when llama.cpp changes underneath
#
# The tensor type table is the part most likely to go stale — it did, three types out of
# date one bump after being written — so it has three tiers: the ggml the image ships,
# this file's table, and sizing by file size. None of them may size a model as free.
# --------------------------------------------------------------------------- #

UNKNOWN_TYPE = 61  # not a ggml type in any build this was written against


def test_the_type_table_matches_the_ggml_that_will_read_the_files():
    """Where libggml can be loaded it wins, so a bump cannot leave the table stale."""
    probed = configure_llamacpp.probe_ggml_types()
    if not probed:
        pytest.skip("no libggml on LD_LIBRARY_PATH (it ships in the image, not in CI)")
    for type_id, (name, block, size) in probed.items():
        if type_id in configure_llamacpp.GGML_TYPES:
            assert configure_llamacpp.GGML_TYPES[type_id] == (name, block, size), f"type {type_id}"


class FakeGgml:
    """libggml as the probe sees it: a type table with gaps, and garbage past its end.

    Past the end, ggml_type_name() returns pointers into whatever follows the table — on
    the board the first ones happened to be mapped and two of them were not, which is the
    segfault the probe must never reach. Here every read past the first is recorded so the
    tests can assert it never happens.
    """

    GAP = (b"DEPRECATED", 0, 0)
    GARBAGE = (bytes([0xFD, 0x7B, 0xBD, 0xA9, 0xF5, 0x0B]), 0, 0)  # as read on the board past id 42

    def __init__(self, types: dict, first_past_end: tuple = GARBAGE):
        self.types = types
        self.end = max(types) + 1
        self.first_past_end = first_past_end
        self.read_past_end = []

    def _entry(self, type_id):
        if type_id in self.types:
            name, block, size = self.types[type_id]
            return name.encode(), block, size
        if type_id < self.end:
            return self.GAP
        if type_id > self.end:
            self.read_past_end.append(type_id)
        return self.first_past_end

    def ggml_type_name(self, type_id):
        return self._entry(type_id)[0]

    def ggml_blck_size(self, type_id):
        return self._entry(type_id)[1]

    def ggml_type_size(self, type_id):
        return self._entry(type_id)[2]


def test_the_probe_stops_at_the_end_of_the_enum_and_never_reads_past_it():
    """The regression behind an 8B model started on one session: probing type ids past
    the enum dereferenced garbage and segfaulted, silently, in a command substitution."""
    ggml = FakeGgml(configure_llamacpp.GGML_TYPES)

    probed = {type_id: (name, block, size) for type_id, name, block, size in configure_llamacpp.probe_ggml_types_of(ggml)}

    assert probed == configure_llamacpp.GGML_TYPES
    assert ggml.read_past_end == []


def test_the_probe_skips_the_gaps_inside_the_enum_and_sees_the_types_after_them():
    types = {**configure_llamacpp.GGML_TYPES, 45: ("q3_0", 32, 14)}  # 43 and 44 removed, 45 added
    del types[41]  # a gap inside the known range, as 4, 5 and 31-33 are
    ggml = FakeGgml(types)

    probed = {type_id: (name, block, size) for type_id, name, block, size in configure_llamacpp.probe_ggml_types_of(ggml)}

    assert 4 not in probed and 41 not in probed
    assert probed[42] == configure_llamacpp.GGML_TYPES[42]
    # The first gap past the last known type reads as the end: what follows is unreachable
    # by design, and a tensor of that type is refused rather than sized as free.
    assert 45 not in probed
    assert ggml.read_past_end == []


def test_the_probe_sees_a_type_added_right_after_the_known_ones():
    types = {**configure_llamacpp.GGML_TYPES, 43: ("q3_0", 32, 14)}
    ggml = FakeGgml(types)

    probed = dict((type_id, (name, block, size)) for type_id, name, block, size in configure_llamacpp.probe_ggml_types_of(ggml))

    assert probed[43] == ("q3_0", 32, 14)
    assert ggml.read_past_end == []


def test_the_probe_runs_in_a_child_and_keeps_what_it_read_before_a_crash(monkeypatch, capsys, tmp_path):
    """A segfault in the child costs the types it had not printed yet, not the sizing."""
    import subprocess

    monkeypatch.setattr(configure_llamacpp, "find_ggml_library", lambda: tmp_path / "libggml-base.so")
    commands = []

    def crashed_child(command, **kwargs):
        commands.append(command)
        stdout = "\t".join(["0", "f32", "1", "4"]) + "\n" + "\t".join(["2", "q4_0", "32", "18"]) + "\n"
        return subprocess.CompletedProcess(command, returncode=-11, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", crashed_child)

    probed = configure_llamacpp.probe_ggml_types()

    assert probed == {0: ("f32", 1, 4), 2: ("q4_0", 32, 18)}
    assert commands == [[configure_llamacpp.sys.executable, configure_llamacpp.__file__, "--probe-ggml-types", str(tmp_path / "libggml-base.so")]]
    assert "libggml probe died with exit status -11" in capsys.readouterr().err


def test_a_library_that_cannot_be_loaded_leaves_the_built_in_table(monkeypatch, capsys):
    """End to end through the real child process: a library that is not there makes the
    child fail, the parent says so, and the sizing runs on the built-in table."""
    monkeypatch.setattr(configure_llamacpp, "find_ggml_library", lambda: Path("/nonexistent/libggml-base.so"))
    monkeypatch.setattr(configure_llamacpp, "_ggml_types", None)

    assert configure_llamacpp.probe_ggml_types() == {}
    assert configure_llamacpp.ggml_types() == configure_llamacpp.GGML_TYPES
    assert "libggml probe died" in capsys.readouterr().err


def test_no_library_on_the_path_means_no_probe(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    assert configure_llamacpp.find_ggml_library() is None
    assert configure_llamacpp.probe_ggml_types() == {}


def test_malformed_probe_output_is_not_a_type():
    lines = [
        ["0", "f32", "1", "4"],
        ["Segmentation fault"],  # not four fields
        ["7", "q5_1", "x", "24"],  # not a number
        ["8", "q8_0", "32", "34", ""],  # five fields
        ["2", "q4_0", "32", "18"],
    ]
    output = "".join("\t".join(fields) + "\n" for fields in lines)

    assert configure_llamacpp.parse_probed_types(output) == {0: ("f32", 1, 4), 2: ("q4_0", 32, 18)}


def test_a_tensor_of_an_unknown_type_is_refused_rather_than_sized_as_free(tmp_path):
    """The dangerous failure: a type whose size is unknown must not be counted as zero
    bytes, which would under-size the model and start the server unable to load it."""
    gguf = write_gguf(
        tmp_path / "m.gguf",
        {"general.architecture": "llama", "llama.block_count": 1},
        [("blk.0.attn_k.weight", (4096,), UNKNOWN_TYPE)],
    )

    with pytest.raises(ValueError, match="unknown ggml tensor type"):
        read_model_shape(gguf)


def test_a_model_that_cannot_be_read_is_sized_by_its_file_size(tmp_path, capsys):
    """The fallback tier: the sizing the runner used before it read headers. Skipping the
    model would be the one unsafe outcome, since the server would then come up with too
    few sessions to load it."""
    small = tmp_path / "small.gguf"
    small.write_bytes(b"GGUF" + bytes(64))  # a plausible magic and nothing else
    models = configure_llamacpp.find_models(tmp_path)

    assert detect_hexagon_sessions(models, 4096) == 1

    diagnostics = capsys.readouterr().err
    assert "cannot size it from its GGUF header" in diagnostics
    assert "falling back to its file size" in diagnostics
    assert "0.00 GB on disk, sized by file size" in diagnostics
    # ... and a big one lands on the tables (sized without a file: see the next test)
    assert configure_llamacpp.FileSizeSizing("big", int(4.8 * 10**9)).sessions_needed(4096) == 3


def test_the_file_size_fallback_reproduces_the_tables_it_came_from():
    """The thresholds the runner shipped before headers were read, unchanged."""

    def sizing(gigabytes):
        return configure_llamacpp.FileSizeSizing("some-model", int(gigabytes * 10**9))

    assert sizing(0.5).sessions_needed(16384) == 1
    assert sizing(2.4).sessions_needed(16384) == 3
    assert sizing(4.8).sessions_needed(16384) == 4
    # the small-context table is more generous, as it was
    assert sizing(2.4).sessions_needed(4096) == 1
    assert sizing(4.8).sessions_needed(4096) == 3
    assert sizing(5.5).sessions_needed(4096) == 4


def test_the_file_size_fallback_keeps_the_gemma_pins(tmp_path, capsys):
    """The tables over-allocate the matformer gemmas by a factor of three, so the sizing
    they shipped with pinned those by name; an unreadable gemma must land there too rather
    than on the table, which would cap the whole server's context for it."""
    sizing = configure_llamacpp.FileSizeSizing
    for name, gigabytes, pinned in (("gemma-4-E2B_q4_0-it", 3.35, 1), ("gemma-4-E4B_q4_0-it", 5.15, 2), ("gemma-4-E4B-Q8_0", 10.5, 2)):
        for ctx_size in CTX_SIZES:
            assert sizing(name, int(gigabytes * 10**9)).sessions_needed(ctx_size) == pinned, name
    # through the runner's own path, on a file whose header cannot be read
    (tmp_path / "gemma-4-E4B_q4_0-it.gguf").write_bytes(b"GGUF" + bytes(64))
    models = configure_llamacpp.find_models(tmp_path)

    assert detect_hexagon_sessions(models, 16384) == 2
    assert detect_ctx_size(models, 16384) == 16384
    assert "pinned by name" in capsys.readouterr().err


def test_a_missing_file_is_ignored_rather_than_guessed(tmp_path, capsys):
    """Nothing can be said about a file that is not there."""
    models = {"gone": {"model": str(tmp_path / "gone.gguf")}}

    assert detect_hexagon_sessions(models, 4096) == 1
    assert "cannot read it at all" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The data type the KV cache is stored in
#
# The cache is f16 unless the service configures otherwise, and a quantized one is
# measurably smaller: Qwen3.5-4B at 16k holds 512 MiB of f16 cache and 272 MiB of q8_0,
# which is the difference between needing two sessions and one.
# --------------------------------------------------------------------------- #


def test_the_kv_cache_is_sized_by_the_configured_cache_type(tmp_path, monkeypatch):
    gguf = attention_model(tmp_path / "m.gguf", layers=32, kv_heads=8, head_size=128)

    f16 = read_model_shape(gguf).kv_cache_bytes(4096)

    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")
    q8_0 = read_model_shape(gguf).kv_cache_bytes(4096)

    assert f16 == 512 * MIB
    # q8_0 stores 32 elements in 34 bytes, so 53% of what f16 costs.
    assert q8_0 == pytest.approx(f16 * 34 / 64, rel=1e-3)


def test_the_k_and_v_caches_are_sized_apart(tmp_path, monkeypatch):
    """llama-server takes a type for each, and a quantized V needs flash attention
    while a quantized K does not, so the two are worth configuring separately."""
    gguf = attention_model(tmp_path / "m.gguf", layers=32, kv_heads=8, head_size=128)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")

    mixed = read_model_shape(gguf).kv_cache_bytes(4096)

    assert mixed == pytest.approx(512 * MIB * (34 / 32 + 2) / 4, rel=1e-3)
    assert configure_llamacpp.kv_cache_description() == "q8_0/f16"


def test_an_unset_cache_type_is_sized_as_f16(tmp_path, monkeypatch):
    """f16 is what llama.cpp stores the cache in when nothing says otherwise."""
    monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_K", raising=False)
    monkeypatch.delenv("LLAMA_ARG_CACHE_TYPE_V", raising=False)
    gguf = attention_model(tmp_path / "m.gguf", layers=32, kv_heads=8, head_size=128)

    assert read_model_shape(gguf).kv_cache_bytes(4096) == 512 * MIB
    assert configure_llamacpp.kv_cache_description() == "f16"


def test_an_unrecognised_cache_type_is_sized_as_the_largest(tmp_path, monkeypatch):
    """Sizing a cache smaller than it turns out to be is what fails a load; sizing it
    larger only costs a session."""
    gguf = attention_model(tmp_path / "m.gguf", layers=32, kv_heads=8, head_size=128)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "something-new")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "something-new")

    assert read_model_shape(gguf).kv_cache_bytes(4096) == 512 * MIB * 2


def test_the_windowed_cache_follows_the_configured_batch_and_slots(tmp_path, monkeypatch):
    """A sliding-window cache holds n_seq_max * window + n_ubatch cells, and both come
    from the environment the server will see: a bigger micro-batch or more slots means a
    bigger cache, and sizing for the defaults would under-count it."""
    gguf = attention_model(tmp_path / "m.gguf", layers=8, kv_heads=1, head_size=256, window=512)
    per_cell = 8 * 1 * 512 * 2

    monkeypatch.setenv("LLAMA_ARG_UBATCH", "1024")
    monkeypatch.setenv("LLAMA_ARG_N_PARALLEL", "8")
    assert read_model_shape(gguf).kv_cache_bytes(16384) == per_cell * (8 * 512 + 1024)

    # Unset, or not a positive number, means llama-server's own defaults.
    monkeypatch.setenv("LLAMA_ARG_UBATCH", "auto")
    monkeypatch.delenv("LLAMA_ARG_N_PARALLEL")
    assert read_model_shape(gguf).kv_cache_bytes(16384) == per_cell * (4 * 512 + 256)


def test_a_q8_0_cache_buys_back_a_session_on_a_kv_heavy_model(tmp_path):
    """The measured case that matters: an 8B model at 16k loads on no number of sessions
    with an f16 cache, and on three with a q8_0 one. Sized for four it would drag the
    whole server's context down (see detect_ctx_size); sized for three it does not."""
    qwen3_8b = dict(npu_weight_mib=3739, n_layer=36, kv_layers=36)
    # 8 KV heads of 128, K and V
    f16 = shape_of(bytes_per_cell=8 * 256 * 2, **qwen3_8b)
    q8_0 = shape_of(bytes_per_cell=int(8 * 256 * 34 / 32), **qwen3_8b)

    assert f16.kv_cache_bytes(16384) == 2304 * MIB
    assert q8_0.kv_cache_bytes(16384) == pytest.approx(1224 * MIB, rel=1e-3)
    assert f16.sessions_needed(16384) == configure_llamacpp.CTX_CAP_MIN_SESSIONS
    assert q8_0.sessions_needed(16384) < configure_llamacpp.CTX_CAP_MIN_SESSIONS


# --------------------------------------------------------------------------- #
# Session sizing
# --------------------------------------------------------------------------- #


def shape_of(*, npu_weight_mib: float, n_layer: int, kv_layers: int, bytes_per_cell: int, recurrent: bool = False) -> ModelShape:
    """A model shape stated in the terms the board measurements are recorded in."""
    return ModelShape(
        n_layer=n_layer,
        npu_weight_bytes=int(npu_weight_mib * MIB),
        kv_layers=tuple([KvLayer(bytes_per_cell, 0)] * kv_layers),
        state_bytes=configure_llamacpp.RECURRENT_STATE_ALLOWANCE if recurrent else 0,
    )


def test_the_busiest_session_holds_one_layer_more_than_an_even_share():
    """Measured on the per-session figures llama-server logs: granite-4.2-3b over two
    sessions puts 21 of its 40 layers' worth of KV cache on HTP0, 672 MiB of 1280."""
    granite = shape_of(npu_weight_mib=1688, n_layer=40, kv_layers=40, bytes_per_cell=2048)

    assert granite.kv_cache_bytes(16384) == 1280 * MIB
    assert granite.session_bytes(16384, 2) == pytest.approx((1688 + 1280) * MIB * 21 / 40, rel=1e-3)
    # One session holds all of it, never more.
    assert granite.session_bytes(16384, 1) == granite.npu_bytes(16384)


def test_sessions_are_added_until_the_busiest_session_fits_the_budget():
    budget = configure_llamacpp.SESSION_BUDGET
    small = shape_of(npu_weight_mib=budget / MIB - 1, n_layer=32, kv_layers=1, bytes_per_cell=0)
    big = shape_of(npu_weight_mib=budget / MIB + 1, n_layer=32, kv_layers=1, bytes_per_cell=0)

    assert small.sessions_needed(4096) == 1
    assert big.sessions_needed(4096) == 2


def test_a_model_too_big_for_every_session_asks_for_them_all():
    huge = shape_of(npu_weight_mib=40000, n_layer=32, kv_layers=1, bytes_per_cell=0)

    assert huge.sessions_needed(16384) == configure_llamacpp.MAX_SESSIONS


# The measured models, as the GGUF header describes them, against the sessions each needed
# at each context: the smallest count that loaded on every attempt (SESSION_ALLOCATION.md),
# None where no count up to four did. The sizing may ask for more than that —
# over-allocating costs a little throughput, under-allocating fails the load — but never
# for less, which is what this pins down.
MEASURED_MODELS = {
    "SmolVLM2-500M-Video-Instruct-Q8_0": (shape_of(npu_weight_mib=366, n_layer=32, kv_layers=32, bytes_per_cell=1280), (1, 1, 1)),
    "Qwen3.5-0.8B-Q4_0": (shape_of(npu_weight_mib=249, n_layer=24, kv_layers=6, bytes_per_cell=2048, recurrent=True), (1, 1, 1)),
    "granite-4.2-3b-Q4_0": (shape_of(npu_weight_mib=1688, n_layer=40, kv_layers=40, bytes_per_cell=2048), (1, 2, 2)),
    "microsoft_Phi-4-mini-instruct-Q4_0": (shape_of(npu_weight_mib=1734, n_layer=32, kv_layers=32, bytes_per_cell=4096), (2, 2, 3)),
    "Qwen3-4B-Instruct-2507-Q4_0": (shape_of(npu_weight_mib=1955, n_layer=36, kv_layers=36, bytes_per_cell=4096), (2, 2, 3)),
    "Qwen3.5-4B-Q4_0-pure": (shape_of(npu_weight_mib=2259, n_layer=32, kv_layers=8, bytes_per_cell=4096, recurrent=True), (1, 1, 2)),
    "Nemotron-Mini-4B-Instruct-Q4_0": (shape_of(npu_weight_mib=1412, n_layer=32, kv_layers=32, bytes_per_cell=4096), (1, 2, 3)),
    "Qwen3.5-4B-Q4_0": (shape_of(npu_weight_mib=1790, n_layer=32, kv_layers=8, bytes_per_cell=4096, recurrent=True), (1, 1, 2)),
    "Qwen3.5-4B-Q8_0": (shape_of(npu_weight_mib=4264, n_layer=32, kv_layers=8, bytes_per_cell=4096, recurrent=True), (2, 2, None)),
    "DeepSeek-R1-Distill-Llama-8B-Q4_0": (shape_of(npu_weight_mib=3759, n_layer=32, kv_layers=32, bytes_per_cell=4096), (2, 3, None)),
    "Qwen_Qwen3-8B-Q4_0": (shape_of(npu_weight_mib=3739, n_layer=36, kv_layers=36, bytes_per_cell=4096), (2, 3, None)),
    "LFM2.5-8B-A1B-Q4_0": (shape_of(npu_weight_mib=4407, n_layer=24, kv_layers=6, bytes_per_cell=2048, recurrent=True), (2, 3, 2)),
    "granite-4.2-8b-Q4_0": (shape_of(npu_weight_mib=4276, n_layer=40, kv_layers=40, bytes_per_cell=4096), (3, 3, 4)),
    "bar-Qwen_Qwen3.5-9B-Q4_0": (shape_of(npu_weight_mib=4018, n_layer=33, kv_layers=9, bytes_per_cell=4096, recurrent=True), (2, 2, 2)),
}


@pytest.mark.parametrize("name", sorted(MEASURED_MODELS))
def test_the_measured_models_never_get_fewer_sessions_than_they_need(name):
    shape, measured = MEASURED_MODELS[name]

    for ctx_size, needed in zip(CTX_SIZES, measured):
        asked = shape.sessions_needed(ctx_size)
        if needed is None:
            # Measured as not loading on any number of sessions at this context: the
            # sizing cannot see that, the context cap is what keeps it out of trouble.
            assert asked == configure_llamacpp.MAX_SESSIONS
        else:
            assert asked >= needed, f"{name} at {ctx_size}: sized for {asked}, needs {needed}"


# The two cells the budget cannot place within a session of their measurement. Giving
# Qwen3.5-4B-Q8_0 three sessions at 16k needs a budget of 1887 MiB, and keeping
# Nemotron-Mini-4B off two sessions at 16k needs one below 1838 MiB, so no single budget
# does both; the safe side of that trade puts Q8_0 on four, which is what the size tables
# it replaced also did. LFM2.5-8B's measured two at 16k rests on a single attempt, against
# a failure on two sessions at 8k.
OVER_ALLOCATED_BY_TWO = {("Qwen3.5-4B-Q8_0", 16384), ("LFM2.5-8B-A1B-Q4_0", 16384)}


def test_the_measured_models_are_not_over_allocated_by_more_than_a_session():
    """The other half of the trade-off: the sizing stays within one session of what was
    measured, so the conservatism costs at most one session's throughput."""
    for name, (shape, measured) in MEASURED_MODELS.items():
        for ctx_size, needed in zip(CTX_SIZES, measured):
            if needed is None:
                continue
            slack = 2 if (name, ctx_size) in OVER_ALLOCATED_BY_TWO else 1
            assert shape.sessions_needed(ctx_size) - needed <= slack, f"{name} at {ctx_size}"


# --------------------------------------------------------------------------- #
# The two answers run-model-router.sh reads
# --------------------------------------------------------------------------- #


def _install(tmp_path, **models) -> dict:
    """Install GGUFs of the given (layers, weight_mib) shapes and index them."""
    for name, (layers, weight_mib) in models.items():
        attention_model(tmp_path / f"{name}.gguf", layers=layers, weight_mib=weight_mib)
    return configure_llamacpp.find_models(tmp_path)


def test_detect_hexagon_sessions_takes_the_hungriest_model(tmp_path, capsys):
    models = _install(tmp_path, small=(32, 100), big=(32, 4000))

    assert detect_hexagon_sessions(models, 4096) == 3

    diagnostics = capsys.readouterr().err
    assert "small: 100 MiB on the NPU" in diagnostics
    assert "fits 1 session" in diagnostics
    assert "requires 3 sessions" in diagnostics


def test_detect_hexagon_sessions_ignores_models_it_cannot_read(tmp_path, capsys):
    models = _install(tmp_path, good=(32, 100))
    (tmp_path / "broken.gguf").write_bytes(b"nonsense")
    models["broken"] = {"model": str(tmp_path / "broken.gguf")}

    assert detect_hexagon_sessions(models, 4096) == 1
    assert "broken: cannot size it from its GGUF header" in capsys.readouterr().err


def test_detect_ctx_size_halves_the_context_until_the_models_fit(tmp_path, capsys):
    """A model that needs every session at 16k gets 8k instead — measured behaviour
    for the 8B models, which load on three sessions at 8k and on none at 16k."""
    models = _install(tmp_path, big=(32, 3739))

    assert detect_ctx_size(models, 16384) == 8192

    diagnostics = capsys.readouterr().err
    assert "needs 4 sessions to hold 16384 tokens" in diagnostics
    assert "capping the context to 8192" in diagnostics


def test_detect_ctx_size_stops_at_the_floor(tmp_path):
    """Nothing is capped below MIN_CTX_SIZE: a model that does not fit there is the
    router's problem, not something a smaller context can fix."""
    models = _install(tmp_path, huge=(32, 12000))

    assert detect_ctx_size(models, 16384) == configure_llamacpp.MIN_CTX_SIZE


def test_detect_ctx_size_keeps_a_context_every_model_holds(tmp_path, capsys):
    models = _install(tmp_path, small=(32, 500), medium=(32, 1500))

    assert detect_ctx_size(models, 16384) == 16384
    assert "every model holds the 16384 token context" in capsys.readouterr().err


@pytest.mark.parametrize("ctx_size", [512, 4096])
def test_a_context_at_or_below_the_floor_is_never_touched(tmp_path, ctx_size):
    models = _install(tmp_path, huge=(32, 12000))

    assert detect_ctx_size(models, ctx_size) == ctx_size


@pytest.mark.parametrize("mode", ["--print-ctx", "--print-ndev"])
def test_an_unconfigured_context_is_taken_as_the_default_one(tmp_path, monkeypatch, capsys, mode):
    """--ctx 0 is "not configured": both answers are computed for DEFAULT_CTX_SIZE, the
    context run-model-router.sh exports in that case, rather than for nothing at all."""
    _install(tmp_path, qwen3_8b=(36, 3739))
    monkeypatch.setattr(configure_llamacpp.sys, "argv", ["configure-llamacpp.py", str(tmp_path), mode, "--ctx", "0"])

    configure_llamacpp.main()

    out, err = capsys.readouterr()
    assert "an unconfigured context (taken as 16384)" in err
    # An 8B model at 16k: the context is halved to 8k, and at 8k it needs 3 sessions.
    assert out.strip() == {"--print-ctx": "8192", "--print-ndev": "4"}[mode]


def test_the_4b_model_set_keeps_the_full_context(tmp_path):
    """A board carrying only 4B-class models holds 16k on two sessions (measured), so
    nothing caps the context there."""
    models = {}
    for name in ("Qwen3.5-4B-Q4_0", "Qwen3.5-4B-Q4_0-pure", "Qwen3.5-0.8B-Q4_0"):
        shape, _ = MEASURED_MODELS[name]
        models[name] = shape

    needed = max(shape.sessions_needed(16384) for shape in models.values())
    assert needed == 2


def test_an_8b_model_trades_the_context_for_sessions(tmp_path):
    """With an 8B model installed the context comes down to 8k, where
    three sessions hold it — where the size tables it replaces cut straight to 4k."""
    models = _install(tmp_path, qwen3_8b=(36, 3739), qwen3_4b=(32, 1790))

    ctx_size = detect_ctx_size(models, 16384)
    assert ctx_size == 8192
    assert detect_hexagon_sessions(models, ctx_size) == 3


# --------------------------------------------------------------------------- #
# Served model names, from the ".arduino_metadata.yaml" download records
#
# Both runner scripts carry the same (duplicated) naming code, so every test runs
# against both: a divergence between the two images is itself a bug.
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["npu-runner", "cpu-runner"])
def runner(request):
    return {"npu-runner": configure_llamacpp, "cpu-runner": configure_llamacpp_cpu}[request.param]


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0")
    return path


def _write_records(directory: Path, *records):
    """A ".arduino_metadata.yaml" the way the models-downloader writes it."""
    lines = ["models:"]
    for origin, files in records:
        lines.append(f"- model_origin: {origin}")
        lines.append(f"  files: [{', '.join(files)}]")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".arduino_metadata.yaml").write_text("\n".join(lines) + "\n")


def test_a_file_without_a_record_keeps_its_stem(runner, tmp_path):
    """The fallback: no ".arduino_metadata.yaml" means an out-of-the-box model."""
    gguf = _gguf(tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf" / "gemma-4-E2B_q4_0-it.gguf")
    assert runner.gguf_model_name(gguf, tmp_path) == "gemma-4-E2B_q4_0-it"


def test_a_curated_download_keeps_its_stem(runner, tmp_path):
    repo = tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    gguf = _gguf(repo / "gemma-4-E2B_q4_0-it.gguf")
    _write_records(repo, ("built_in", ["gemma-4-E2B_q4_0-it.gguf"]))
    assert runner.gguf_model_name(gguf, tmp_path) == "gemma-4-E2B_q4_0-it"


def test_a_user_download_is_named_by_its_path(runner, tmp_path):
    """Ad-hoc downloads carry the repository in the name, so two same-named files
    from different owners never collide (mirrors the models-downloader listing id)."""
    repo = tmp_path / "unsloth" / "Qwen3-0.6B-GGUF"
    gguf = _gguf(repo / "Qwen3-0.6B-Q4_0.gguf")
    _write_records(repo, ("user", ["Qwen3-0.6B-Q4_0.gguf"]))
    assert runner.gguf_model_name(gguf, tmp_path) == "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0"


def test_the_record_is_found_above_a_nested_quantization_folder(runner, tmp_path):
    """Some repositories nest their files per quantization; the record stays at the
    repository directory and lists the nested relative path."""
    repo = tmp_path / "org" / "repo-GGUF"
    gguf = _gguf(repo / "Q4_0" / "model.gguf")
    _write_records(repo, ("user", ["Q4_0/model.gguf"]))
    assert runner.gguf_model_name(gguf, tmp_path) == "org/repo-GGUF/Q4_0/model"


def test_each_quantization_answers_to_its_own_record(runner, tmp_path):
    """A curated and an ad-hoc quantization share the repository directory."""
    repo = tmp_path / "unsloth" / "Qwen3-0.6B-GGUF"
    q4 = _gguf(repo / "Qwen3-0.6B-Q4_0.gguf")
    q8 = _gguf(repo / "Qwen3-0.6B-Q8_0.gguf")
    _write_records(
        repo,
        ("built_in", ["Qwen3-0.6B-Q4_0.gguf"]),
        ("user", ["Qwen3-0.6B-Q8_0.gguf"]),
    )
    assert runner.gguf_model_name(q4, tmp_path) == "Qwen3-0.6B-Q4_0"
    assert runner.gguf_model_name(q8, tmp_path) == "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0"


def test_an_unclaimed_sibling_counts_as_out_of_the_box(runner, tmp_path):
    """A GGUF no record names (installed by something that recorded nothing) falls
    back to the out-of-the-box naming, never to a sibling's record."""
    repo = tmp_path / "unsloth" / "Qwen3-0.6B-GGUF"
    q8 = _gguf(repo / "Qwen3-0.6B-Q8_0.gguf")
    _write_records(repo, ("user", ["Qwen3-0.6B-Q4_0.gguf"]))
    assert runner.gguf_model_name(q8, tmp_path) == "Qwen3-0.6B-Q8_0"


def test_unreadable_metadata_degrades_to_out_of_the_box(runner, tmp_path):
    repo = tmp_path / "unsloth" / "Qwen3-0.6B-GGUF"
    gguf = _gguf(repo / "Qwen3-0.6B-Q4_0.gguf")
    repo.joinpath(".arduino_metadata.yaml").write_text("a: b: c\n")
    assert runner.gguf_model_name(gguf, tmp_path) == "Qwen3-0.6B-Q4_0"


def test_models_ini_serves_each_model_under_its_derived_name(runner, tmp_path, capsys):
    """End to end: one curated and one ad-hoc install become two sections, and the
    mmproj companion in the ad-hoc repository is attached to its model."""
    curated = tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    _gguf(curated / "gemma-4-E2B_q4_0-it.gguf")
    _write_records(curated, ("built_in", ["gemma-4-E2B_q4_0-it.gguf"]))
    adhoc = tmp_path / "unsloth" / "gemma-4-E4B-it-GGUF"
    _gguf(adhoc / "gemma-4-E4B-it-Q4_0.gguf")
    _gguf(adhoc / "mmproj-BF16.gguf")
    _write_records(adhoc, ("user", ["gemma-4-E4B-it-Q4_0.gguf", "mmproj-BF16.gguf"]))

    runner.generate_models_ini(tmp_path)

    config = configparser.ConfigParser()
    config.read(tmp_path / "models.ini")
    assert sorted(config.sections()) == [
        "gemma-4-E2B_q4_0-it",
        "unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0",
    ]
    assert config["unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0"]["mmproj"].endswith("mmproj-BF16.gguf")
