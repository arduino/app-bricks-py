# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the llama.cpp runners' configure-llamacpp.py scripts.

Covers the Hexagon session sizing of the NPU runner, and — for both runners, whose
scripts deliberately duplicate the code — the served model names, which are derived
from the ".arduino_metadata.yaml" download records rather than from a catalog baked
into the images.
"""

from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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

model_ndev = configure_llamacpp.model_ndev
detect_hexagon_ndev = configure_llamacpp.detect_hexagon_ndev
detect_ctx_size = configure_llamacpp.detect_ctx_size

GB = int(1e9)

# The context sizes the runner sizes for: 0 (unknown), the default 16k, and the capped 4k.
ALL_CTX_SIZES = (0, 16384, 4096)


@pytest.mark.parametrize("ctx_size", ALL_CTX_SIZES)
@pytest.mark.parametrize(
    "name, gguf_gb",
    [
        ("gemma-4-E2B_q4_0-it", 3.35),
        ("gemma-4-E2B-Q8_0", 6.9),
        ("google_gemma-4-E2B-it-qat-Q4_0", 3.35),
    ],
)
def test_e2b_pinned_to_one_session(name, gguf_gb, ctx_size):
    """E2B is pinned to a single session at every context size and quantization."""
    assert model_ndev(name, int(gguf_gb * GB), ctx_size) == 1


@pytest.mark.parametrize("ctx_size", ALL_CTX_SIZES)
@pytest.mark.parametrize(
    "name, gguf_gb",
    [
        ("gemma-4-E4B_q4_0-it", 5.15),
        ("gemma-4-E4B-Q8_0", 10.5),
        ("google_gemma-4-E4B-it-qat-Q4_0", 5.15),
    ],
)
def test_e4b_pinned_to_two_sessions(name, gguf_gb, ctx_size):
    """E4B is pinned to two sessions at every context size and quantization."""
    assert model_ndev(name, int(gguf_gb * GB), ctx_size) == 2


@pytest.mark.parametrize(
    "name, gguf_gb, ctx_size, expected",
    [
        # Default profile: sized by GGUF size, the pins do not leak onto other models.
        ("Qwen3.5-0.8B-Q4_0", 0.51, 16384, 1),
        ("Qwen3.5-4B-Q4_0", 2.78, 16384, 2),
        ("gemma-4-12b-it-Q4_0", 6.98, 16384, 4),
        # Small-context profile: the 1-session measurements still hold; everything bigger
        # takes all 4 sessions since the September 2025 llama.cpp build (Qwen3-8B fails to
        # fastrpc-map its per-domain buffer on 2).
        ("Qwen3.5-0.8B-Q4_0", 0.51, 4096, 1),
        ("Qwen3-4B-2507-Q4_0", 2.38, 4096, 1),
        ("Qwen3.5-4B-Q4_0", 2.78, 4096, 1),
        ("Qwen3-8B-Q4_0", 4.79, 4096, 4),
        ("Qwen3.5-9B-Q4_0", 5.74, 4096, 4),
        ("gemma-4-12b-it-Q4_0", 6.98, 4096, 4),
    ],
)
def test_unpinned_models_are_sized_by_gguf_size(name, gguf_gb, ctx_size, expected):
    assert model_ndev(name, int(gguf_gb * GB), ctx_size) == expected


def _install_models(monkeypatch, sizes_gb: dict[str, float]) -> dict[str, dict]:
    """Return the models mapping for GGUFs of the given sizes, without writing GB to disk."""
    sizes = {name: int(gguf_gb * GB) for name, gguf_gb in sizes_gb.items()}

    class FakePath:
        def __init__(self, path: str):
            self._path = path

        def stat(self):
            if self._path not in sizes:
                raise OSError(2, "No such file or directory")
            return SimpleNamespace(st_size=sizes[self._path])

    monkeypatch.setattr(configure_llamacpp, "Path", FakePath)
    return {name: {"model": name} for name in sizes_gb}


def test_detect_hexagon_ndev_does_not_trigger_the_context_cap(monkeypatch, capsys):
    """The set of models from the ventunoq board needs 2 sessions, not 4.

    Four sessions is what makes run-model-router.sh cap the context to 4k, so pinning E4B to
    two sessions has to keep the whole set below that threshold at the default context.
    """
    models = _install_models(
        monkeypatch,
        {
            "Qwen3.5-4B-Q4_0-pure": 2.38,
            "Qwen_Qwen3.5-4B-Q4_0": 2.78,
            "gemma-4-E2B_q4_0-it": 3.35,
            "gemma-4-E4B_q4_0-it": 5.15,
        },
    )

    assert detect_hexagon_ndev(models, 16384) == 2

    diagnostics = capsys.readouterr().err
    assert "gemma-4-E2B_q4_0-it: 3.35 GB, pinned to 1 session" in diagnostics
    assert "gemma-4-E4B_q4_0-it: 5.15 GB, pinned to 2 sessions" in diagnostics


@pytest.mark.parametrize(
    "name, gguf_gb",
    [
        ("gemma-4-E2B_q4_0-it", 3.35),
        ("gemma-4-E4B_q4_0-it", 5.15),
        ("google_gemma-4-E4B-it-qat-Q8_0", 10.5),
    ],
)
def test_gemma_never_caps_the_context(monkeypatch, capsys, name, gguf_gb):
    """E2B and E4B keep the full context whatever their size."""
    models = _install_models(monkeypatch, {name: gguf_gb})

    assert detect_ctx_size(models, 16384) == 16384
    assert f"{name}: {gguf_gb:.2f} GB, exempt from the context cap" in capsys.readouterr().err


def test_the_ventunoq_model_set_keeps_the_full_context(monkeypatch):
    models = _install_models(
        monkeypatch,
        {
            "Qwen3.5-4B-Q4_0-pure": 2.38,
            "Qwen_Qwen3.5-4B-Q4_0": 2.78,
            "gemma-4-E2B_q4_0-it": 3.35,
            "gemma-4-E4B_q4_0-it": 5.15,
        },
    )

    assert detect_ctx_size(models, 16384) == 16384


def test_qwen3_8b_gets_four_sessions_at_the_capped_context(monkeypatch, capsys):
    """Regression for the 21q board set: Qwen3-8B caps the context to 4k, and at 4k it
    fails to load on 2 sessions (fastrpc_mmap error on the HTP0 buffer with the September
    2025 llama.cpp build), so the whole set has to come up with 4."""
    models = _install_models(
        monkeypatch,
        {
            "Qwen3-0.6B-Q3_K_S": 0.32,
            "Qwen3-0.6B-Q4_0": 0.38,
            "Qwen3.5-4B-Q4_0-pure": 2.38,
            "Qwen_Qwen3-8B-Q4_0": 4.79,
            "Qwen_Qwen3.5-4B-Q4_0": 2.78,
            "gemma-3-1b-it-Q4_0": 0.72,
            "gemma-4-E2B_q4_0-it": 3.35,
            "gemma-4-E4B_q4_0-it": 5.15,
        },
    )

    assert detect_ctx_size(models, 16384) == 4096
    assert detect_hexagon_ndev(models, 4096) == 4
    assert "Qwen_Qwen3-8B-Q4_0: 4.79 GB, requires 4 sessions" in capsys.readouterr().err


def test_a_big_non_exempt_model_still_caps_the_context(monkeypatch, capsys):
    """The context is server-wide: a big model caps it even next to the exempt gemmas."""
    models = _install_models(monkeypatch, {"gemma-4-E4B_q4_0-it": 5.15, "gemma-4-12b-it-Q4_0": 6.98})

    assert detect_ctx_size(models, 16384) == 4096
    assert "gemma-4-12b-it-Q4_0: 6.98 GB, needs 4 sessions" in capsys.readouterr().err


@pytest.mark.parametrize("ctx_size", [0, 512, 4096])
def test_a_context_at_or_below_the_cap_is_never_touched(monkeypatch, ctx_size):
    """0 means "not configured": llama-server picks the context, so there is nothing to cap."""
    models = _install_models(monkeypatch, {"gemma-4-12b-it-Q4_0": 6.98})

    assert detect_ctx_size(models, ctx_size) == ctx_size


def test_detect_ctx_size_ignores_unreadable_models(monkeypatch, capsys):
    models = _install_models(monkeypatch, {"gemma-4-12b-it-Q4_0": 6.98})
    # Sorts before the model that caps, so the scan has to get past it.
    models["a-missing-model"] = {"model": "a-missing-model"}

    assert detect_ctx_size(models, 16384) == 4096
    assert "a-missing-model: cannot read size" in capsys.readouterr().err


def test_detect_hexagon_ndev_ignores_unreadable_models(monkeypatch, capsys):
    models = _install_models(monkeypatch, {"gemma-4-E4B_q4_0-it": 5.15})
    models["missing-model"] = {"model": "missing-model"}

    assert detect_hexagon_ndev(models, 4096) == 2
    assert "missing-model: cannot read size" in capsys.readouterr().err


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
