# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the Hexagon session sizing of the llama.cpp NPU runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "containers" / "ai" / "llamacpp-npu-runner" / "scripts" / "configure-llamacpp.py"

# The script is not importable by name (it lives outside a package and has a dash in it).
_spec = importlib.util.spec_from_file_location("configure_llamacpp", SCRIPT)
configure_llamacpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(configure_llamacpp)

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
        # Small-context profile: reproduces the measurements in the table.
        ("Qwen3.5-0.8B-Q4_0", 0.51, 4096, 1),
        ("Qwen3-4B-2507-Q4_0", 2.38, 4096, 1),
        ("Qwen3.5-4B-Q4_0", 2.78, 4096, 1),
        ("Qwen3-8B-Q4_0", 4.79, 4096, 2),
        ("Qwen3.5-9B-Q4_0", 5.74, 4096, 2),
        ("gemma-4-12b-it-Q4_0", 6.98, 4096, 3),
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
