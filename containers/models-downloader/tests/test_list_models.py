# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``list_models.py``."""

import json
import os

import pytest

import list_models
from common.download_marker import write_marker


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_gguf(path, size_bytes=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size_bytes)


# --------------------------------------------------------------------------- #
# get_model_subdir
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "repository, expected",
    [
        ("/var/lib/arduino-app-cli/models/audio-analytics/tts", "audio-analytics/tts"),
        ("/var/lib/arduino-app-cli/models/genai", "genai"),
        ("models/genai", "genai"),
        ("models/audio-analytics/asr", "audio-analytics/asr"),
        ("llamacpp", "llamacpp"),
        ("edge-impulse", "edge-impulse"),
        ("", ""),
    ],
)
def test_get_model_subdir(repository, expected):
    assert list_models.get_model_subdir(repository) == expected


# --------------------------------------------------------------------------- #
# build_model_directory
# --------------------------------------------------------------------------- #
def test_build_model_directory_full():
    variables = {
        "model_name": "yolo",
        "model_type": "object_detection",
        "quantization": "w8a8",
        "chipset": "qcs6490",
    }
    assert list_models.build_model_directory(variables) == "yolo-object_detection-w8a8-qcs6490"


def test_build_model_directory_incomplete_returns_empty():
    assert list_models.build_model_directory({"model_name": "yolo"}) == ""
    assert list_models.build_model_directory({}) == ""


# --------------------------------------------------------------------------- #
# get_dir_size_mb
# --------------------------------------------------------------------------- #
def test_get_dir_size_mb_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * (2 * 1024 * 1024))  # 2 MB
    assert list_models.get_dir_size_mb(str(f)) == 2.0


def test_get_dir_size_mb_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\0" * (1024 * 1024))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"\0" * (1024 * 1024))
    assert list_models.get_dir_size_mb(str(tmp_path)) == 2.0


def test_get_dir_size_mb_missing_returns_none(tmp_path):
    assert list_models.get_dir_size_mb(str(tmp_path / "nope")) is None


# --------------------------------------------------------------------------- #
# check_model_exists
# --------------------------------------------------------------------------- #
def test_check_model_exists_no_model_directory():
    exists, path = list_models.check_model_exists({"model_directory": ""}, "/models")
    assert exists is False
    assert path == ""


def test_check_model_exists_exact_match(tmp_path):
    base = tmp_path / "models"
    (base / "edge-impulse" / "my-model").mkdir(parents=True)
    model_info = {"model_directory": "my-model", "models_repository": "edge-impulse"}
    exists, path = list_models.check_model_exists(model_info, str(base))
    assert exists is True
    assert path == os.path.join(str(base), "edge-impulse", "my-model")


def test_check_model_exists_fuzzy_underscore_match(tmp_path):
    base = tmp_path / "models"
    # On disk the folder uses underscores; the YAML model_directory uses hyphens.
    (base / "my_model_proxy").mkdir(parents=True)
    model_info = {"model_directory": "my-model", "models_repository": ""}
    exists, path = list_models.check_model_exists(model_info, str(base))
    assert exists is True
    assert path.endswith("my_model_proxy")


def test_check_model_exists_missing(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    model_info = {"model_directory": "absent", "models_repository": ""}
    exists, path = list_models.check_model_exists(model_info, str(base))
    assert exists is False


# --------------------------------------------------------------------------- #
# model_is_downloading
# --------------------------------------------------------------------------- #
def test_model_is_downloading_marker_present(tmp_path):
    base = tmp_path / "models"
    model_dir = base / "llamacpp" / "google" / "gemma-gguf"
    model_dir.mkdir(parents=True)
    write_marker(str(model_dir), handler="hf-handler")
    model_info = {"model_directory": "google/gemma-gguf", "models_repository": "llamacpp"}
    data = list_models.model_is_downloading(model_info, str(base))
    assert data is not None
    assert data["status"] == "downloading"


def test_model_is_downloading_absent(tmp_path):
    base = tmp_path / "models"
    (base / "llamacpp" / "google" / "gemma-gguf").mkdir(parents=True)
    model_info = {"model_directory": "google/gemma-gguf", "models_repository": "llamacpp"}
    assert list_models.model_is_downloading(model_info, str(base)) is None


# --------------------------------------------------------------------------- #
# llamacpp_name_from_marker
# --------------------------------------------------------------------------- #
def test_name_from_marker_gguf_url():
    marker = {
        "model_url": "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf"
    }
    assert list_models.llamacpp_name_from_marker(marker, "/models/x") == "gemma-4-E2B_q4_0-it"


def test_name_from_marker_non_gguf_url():
    marker = {"model_url": "https://example.com/path/some-model"}
    assert list_models.llamacpp_name_from_marker(marker, "/models/x") == "some-model"


def test_name_from_marker_model_directory_fallback():
    marker = {"model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf"}
    assert list_models.llamacpp_name_from_marker(marker, "/models/x") == "gemma-4-E2B-it-qat-q4_0-gguf"


def test_name_from_marker_root_fallback():
    marker = {}
    root = os.path.join("models", "llamacpp", "leftover")
    assert list_models.llamacpp_name_from_marker(marker, root) == "leftover"


# --------------------------------------------------------------------------- #
# find_llamacpp_models
# --------------------------------------------------------------------------- #
def test_find_llamacpp_single_model(tmp_path):
    base = tmp_path / "models"
    _make_gguf(str(base / "llamacpp" / "repo" / "model-a.gguf"))
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    entry = results[0]
    assert entry["id"] == "llamacpp:model-a"
    assert entry["name"] == "model-a"
    assert entry["handler"] == "llamacpp"
    assert entry["installed"] is True
    assert entry["downloading"] is False
    assert "mmproj" not in entry


def test_find_llamacpp_groups_mmproj(tmp_path):
    base = tmp_path / "models"
    repo = base / "llamacpp" / "moondream" / "moondream2-gguf"
    _make_gguf(str(repo / "moondream2-text-model-f16.gguf"), size_bytes=1024 * 1024)
    _make_gguf(str(repo / "moondream2-mmproj-f16.gguf"), size_bytes=512 * 1024)
    results = list_models.find_llamacpp_models(str(base))
    # The mmproj file is grouped into the text model, not listed separately.
    assert len(results) == 1
    entry = results[0]
    assert entry["name"] == "moondream2-text-model-f16"
    assert entry["mmproj"].endswith("moondream2-mmproj-f16.gguf")
    # disk size is the sum of both files (1 MB + 0.5 MB).
    assert entry["disk_size_mb"] == 1.5


def test_find_llamacpp_downloading_marker(tmp_path):
    base = tmp_path / "models"
    repo = base / "llamacpp" / "repo" / "model-gguf"
    _make_gguf(str(repo / "model-b.gguf"))
    write_marker(str(repo), handler="hf-handler")
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["installed"] is False
    assert results[0]["downloading"] is True


def test_find_llamacpp_empty_folder_with_marker(tmp_path):
    base = tmp_path / "models"
    repo = base / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    repo.mkdir(parents=True)
    write_marker(
        str(repo),
        handler="hf-handler",
        model_url="https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf",
    )
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    entry = results[0]
    # Name comes from the .gguf filename in the marker url.
    assert entry["id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert entry["name"] == "gemma-4-E2B_q4_0-it"
    assert entry["installed"] is False
    assert entry["downloading"] is True


def test_find_llamacpp_no_directory(tmp_path):
    assert list_models.find_llamacpp_models(str(tmp_path / "models")) == []


# --------------------------------------------------------------------------- #
# get_model_info
# --------------------------------------------------------------------------- #
def test_get_model_info_platform_entry():
    entry = {
        "llamacpp:gemma-4-E2B_q4_0-it": {
            "name": "Gemma 4 E2B",
            "supported_boards": ["ventunoq"],
            "metadata": {"model_size_mb": 3430},
            "deployment": {
                "handler": "hf-handler",
                "platforms": [
                    {
                        "ventunoq": {
                            "variables": {
                                "model_url": "https://hf/repo/gemma.gguf",
                                "models_repository": "llamacpp",
                                "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf",
                            }
                        }
                    }
                ],
            },
        }
    }
    results = list_models.get_model_info(entry)
    assert len(results) == 1
    info = results[0]
    assert info["id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert info["name"] == "Gemma 4 E2B"
    assert info["handler"] == "hf-handler"
    assert info["model_directory"] == "google/gemma-4-E2B-it-qat-q4_0-gguf"
    assert info["models_repository"] == "llamacpp"
    assert info["model_size_mb"] == 3430
    assert info["pre_loaded"] is False


def test_get_model_info_pre_loaded():
    entry = {
        "builtin:asr": {
            "name": "Builtin ASR",
            "deployment": {"handler": "asr-handler", "pre-loaded": True},
        }
    }
    results = list_models.get_model_info(entry)
    assert len(results) == 1
    assert results[0]["pre_loaded"] is True
    assert results[0]["handler"] == "asr-handler"


def test_get_model_info_no_deployment_skipped():
    entry = {"x:y": {"name": "No deployment"}}
    assert list_models.get_model_info(entry) == []


# --------------------------------------------------------------------------- #
# main() integration
# --------------------------------------------------------------------------- #
SAMPLE_YAML = """\
models:
 - "llamacpp:gemma-4-E2B_q4_0-it":
    name: "Gemma 4 E2B"
    supported_boards: ["ventunoq"]
    deployment:
      handler: "hf-handler"
      platforms:
        - ventunoq:
            variables:
              model_url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf"
              models_repository: "llamacpp"
              model_directory: "google/gemma-4-E2B-it-qat-q4_0-gguf"
    metadata:
      model_size_mb: 3430
 - "ei:other-model":
    name: "Other"
    supported_boards: ["ventunoq"]
    deployment:
      handler: "ei-handler"
      platforms:
        - ventunoq:
            variables:
              models_repository: "models/edge-impulse"
              model_name: "other-model.eim"
"""


def _run_main(monkeypatch, capsys, tmp_path, yaml_text, extra_args=None):
    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text(yaml_text)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    argv = [
        "list_models.py",
        "--models-dir",
        str(models_dir),
        "--model-list",
        str(yaml_path),
        "--json",
    ]
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr("sys.argv", argv)
    list_models.main()
    out = capsys.readouterr().out
    return models_dir, json.loads(out)["models"]


def test_main_dedup_merges_filesystem_into_yaml(monkeypatch, capsys, tmp_path):
    """A llamacpp gguf on disk must not duplicate its models-list.yaml entry."""
    models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    # Place the gguf the YAML path check can't resolve (nested model_directory).
    _make_gguf(
        str(models_dir / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf" / "gemma-4-E2B_q4_0-it.gguf")
    )

    list_models._SEARCH_DIR_CACHE.clear()
    models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)

    gemma = [m for m in models if m["id"] == "llamacpp:gemma-4-E2B_q4_0-it"]
    assert len(gemma) == 1
    entry = gemma[0]
    # Canonical YAML metadata is kept...
    assert entry["name"] == "Gemma 4 E2B"
    assert entry["handler"] == "hf-handler"
    assert entry["model_size_mb"] == 3430
    # ...with filesystem-derived installed status.
    assert entry["installed"] is True
    assert entry["downloading"] is False
    assert entry["path"].endswith("gemma-4-E2B_q4_0-it.gguf")


def test_main_not_installed_when_no_files(monkeypatch, capsys, tmp_path):
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    gemma = [m for m in models if m["id"] == "llamacpp:gemma-4-E2B_q4_0-it"]
    assert len(gemma) == 1
    assert gemma[0]["installed"] is False


def test_main_supported_board_filter(monkeypatch, capsys, tmp_path):
    yaml_text = SAMPLE_YAML + """\
 - "ei:unoq-only":
    name: "Uno Q only"
    supported_boards: ["unoq"]
    deployment:
      handler: "ei-handler"
      platforms:
        - unoq:
            variables:
              models_repository: "models/edge-impulse"
              model_name: "unoq-only.eim"
"""
    _models_dir, models = _run_main(
        monkeypatch, capsys, tmp_path, yaml_text, extra_args=["--supported-board", "ventunoq"]
    )
    ids = {m["id"] for m in models}
    assert "ei:unoq-only" not in ids
    assert "llamacpp:gemma-4-E2B_q4_0-it" in ids


def test_main_installed_only_filter(monkeypatch, capsys, tmp_path):
    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text(SAMPLE_YAML)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _make_gguf(
        str(models_dir / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf" / "gemma-4-E2B_q4_0-it.gguf")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "list_models.py",
            "--models-dir",
            str(models_dir),
            "--model-list",
            str(yaml_path),
            "--json",
            "--installed-only",
        ],
    )
    list_models.main()
    models = json.loads(capsys.readouterr().out)["models"]
    assert models
    assert all(m["installed"] for m in models)


def test_main_missing_yaml_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["list_models.py", "--model-list", str(tmp_path / "nope.yaml"), "--json"],
    )
    with pytest.raises(SystemExit) as exc:
        list_models.main()
    assert exc.value.code == 1
