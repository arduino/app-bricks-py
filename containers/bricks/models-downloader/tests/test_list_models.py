# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``list_models.py``."""

import json
import os

import pytest

import list_models
from common.download_marker import write_marker
from common.model_metadata import METADATA_NAME, write_metadata


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_gguf(path, size_bytes=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size_bytes)


def _write_metadata_file(directory, text):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, METADATA_NAME), "w") as f:
        f.write(text)


def _record_install(repo_dir, origin, files, model_id="llamacpp:whatever"):
    """The ".arduino_metadata.yaml" record a download of *files* leaves in *repo_dir*."""
    write_metadata(
        str(repo_dir),
        "hf-handler",
        env={},
        models_list_path="",
        identity={"model_id": model_id, "model_origin": origin},
        files=files,
    )


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
    model_dir = base / "edge-impulse" / "my-model"
    model_dir.mkdir(parents=True)
    (model_dir / "my-model.eim").write_bytes(b"\0")
    model_info = {"model_directory": "my-model", "models_repository": "edge-impulse"}
    exists, path = list_models.check_model_exists(model_info, str(base))
    assert exists is True
    assert path == os.path.join(str(base), "edge-impulse", "my-model")


def test_check_model_exists_fuzzy_underscore_match(tmp_path):
    base = tmp_path / "models"
    # On disk the folder uses underscores; the YAML model_directory uses hyphens.
    model_dir = base / "my_model_proxy"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"\0")
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
    marker = {"model_url": "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf"}
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
    """An ad-hoc model (user-configured record) is named by its path — stable, and
    unique by construction."""
    base = tmp_path / "models"
    repo = base / "llamacpp" / "repo"
    _make_gguf(str(repo / "model-a.gguf"))
    _record_install(repo, "user", ["model-a.gguf"])
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    entry = results[0]
    assert entry["id"] == "llamacpp:repo/model-a"
    assert entry["name"] == "repo/model-a"
    assert entry["handler"] == "llamacpp"
    assert entry["installed"] is True
    assert entry["downloading"] is False
    assert "mmproj" not in entry


def test_find_llamacpp_recordless_model_keeps_its_stem_name(tmp_path):
    """The fallback: a GGUF with no download record is an out-of-the-box model."""
    base = tmp_path / "models"
    _make_gguf(str(base / "llamacpp" / "repo" / "model-a.gguf"))
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["id"] == "llamacpp:model-a"
    assert results[0]["name"] == "model-a"


def test_find_llamacpp_builtin_record_keeps_its_stem_name(tmp_path):
    """A curated download serves under the stem its fixed entry id uses."""
    base = tmp_path / "models"
    repo = base / "llamacpp" / "repo"
    _make_gguf(str(repo / "model-a.gguf"))
    _record_install(repo, "built_in", ["model-a.gguf"], model_id="llamacpp:model-a")
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["id"] == "llamacpp:model-a"
    assert results[0]["name"] == "model-a"


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
    # Ad-hoc pending download: the .gguf filename from the marker url, qualified by
    # location — the same id the finished install will get.
    assert entry["id"] == "llamacpp:google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it"
    assert entry["name"] == "google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it"
    assert entry["installed"] is False
    assert entry["downloading"] is True


def test_marker_covers_only_the_files_it_names():
    marker = {"file_patterns": ["*Q3_K_S*.gguf"]}
    assert list_models.marker_covers_file(marker, "Qwen3-0.6B-Q3_K_S.gguf") is True
    assert list_models.marker_covers_file(marker, "Qwen3-0.6B-Q4_0.gguf") is False
    # No field (another handler, or a marker written before it existed): the whole folder.
    assert list_models.marker_covers_file({}, "Qwen3-0.6B-Q4_0.gguf") is True
    assert list_models.marker_covers_file({"file_patterns": "*.gguf"}, "Qwen3-0.6B-Q4_0.gguf") is True


def test_find_llamacpp_marker_leaves_another_quantization_installed(tmp_path):
    """One repository directory holds every quantization downloaded from that repository."""
    base = tmp_path / "models"
    repo = base / "llamacpp" / "unsloth" / "Qwen3-0.6B-GGUF"
    _make_gguf(str(repo / "Qwen3-0.6B-Q4_0.gguf"))
    _record_install(repo, "user", ["Qwen3-0.6B-Q4_0.gguf"])
    write_marker(
        str(repo),
        handler="hf-handler",
        model_url="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q3_K_S.gguf",
        file_patterns=["*Q3_K_S*.gguf"],
    )

    results = {entry["id"]: entry for entry in list_models.find_llamacpp_models(str(base))}
    installed = results["llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0"]
    assert installed["installed"] is True
    assert installed["downloading"] is False
    # The one on its way is still surfaced from the marker, next to the installed one.
    pending = results["llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q3_K_S"]
    assert pending["installed"] is False
    assert pending["downloading"] is True


def test_find_llamacpp_marker_for_a_gguf_on_disk_lists_it_once(tmp_path):
    base = tmp_path / "models"
    repo = base / "llamacpp" / "unsloth" / "Qwen3-0.6B-GGUF"
    _make_gguf(str(repo / "Qwen3-0.6B-Q3_K_S.gguf"))
    write_marker(str(repo), handler="hf-handler", file_patterns=["*Q3_K_S*.gguf"])

    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["id"] == "llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q3_K_S"
    assert results[0]["downloading"] is True


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
    _make_gguf(str(models_dir / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf" / "gemma-4-E2B_q4_0-it.gguf"))

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


# --------------------------------------------------------------------------- #
# _has_model_content
# --------------------------------------------------------------------------- #
def test_check_model_exists_ignores_metadata_only_dir(tmp_path):
    base = tmp_path / "models"
    _write_metadata_file(str(base / "edge-impulse" / "my-model"), "handler: ei-handler\n")
    model_info = {"model_directory": "my-model", "models_repository": "edge-impulse"}
    exists, _path = list_models.check_model_exists(model_info, str(base))
    assert exists is False


def test_check_model_exists_ignores_marker_only_dir(tmp_path):
    base = tmp_path / "models"
    model_dir = base / "edge-impulse" / "my-model"
    model_dir.mkdir(parents=True)
    write_marker(str(model_dir), handler="ei-handler")
    model_info = {"model_directory": "my-model", "models_repository": "edge-impulse"}
    exists, _path = list_models.check_model_exists(model_info, str(base))
    assert exists is False


def test_check_model_exists_matches_a_file(tmp_path):
    base = tmp_path / "models"
    base.mkdir()
    (base / "model.eim").write_bytes(b"\0")
    model_info = {"model_directory": "model.eim", "models_repository": ""}
    exists, _path = list_models.check_model_exists(model_info, str(base))
    assert exists is True


# --------------------------------------------------------------------------- #
# model_metadata / outdated_fields
# --------------------------------------------------------------------------- #
AI_HUB_INFO = {
    "id": "genie:x",
    "model_directory": "qwen-genie-w4a16",
    "models_repository": "models/audio-analytics/tts",
    "variables": {"model_directory": "qwen-genie-w4a16", "version": "0.57.1"},
}


def test_model_metadata_reads_from_nested_repository(tmp_path):
    base = tmp_path / "models"
    _write_metadata_file(str(base / "audio-analytics" / "tts" / "qwen-genie-w4a16"), "models:\n- model_id: genie:x\n")
    data = list_models.model_metadata(AI_HUB_INFO, str(base))
    assert data["model_id"] == "genie:x"


def test_model_metadata_absent_returns_none(tmp_path):
    assert list_models.model_metadata(AI_HUB_INFO, str(tmp_path / "models")) is None


def test_model_metadata_falls_back_to_the_matched_path(tmp_path):
    base = tmp_path / "models"
    # The folder on disk carries a "_proxy" suffix, so the canonical path misses it.
    matched = base / "qwen_genie_w4a16_proxy"
    _write_metadata_file(str(matched), "models:\n- model_id: genie:fuzzy\n")
    model_info = {"id": "genie:fuzzy", "model_directory": "qwen-genie-w4a16", "models_repository": ""}
    data = list_models.model_metadata(model_info, str(base), path=str(matched))
    assert data["model_id"] == "genie:fuzzy"


def test_outdated_fields_detects_changed_variables():
    metadata = {"inputs": {"model_directory": "qwen-genie-w4a16", "version": "0.51.0"}}
    assert list_models.outdated_fields(AI_HUB_INFO, metadata) == ["version"]


def test_outdated_fields_empty_when_inputs_match():
    metadata = {"inputs": {"model_directory": "qwen-genie-w4a16", "version": "0.57.1"}}
    assert list_models.outdated_fields(AI_HUB_INFO, metadata) == []


def test_outdated_fields_reports_missing_inputs():
    assert list_models.outdated_fields(AI_HUB_INFO, {}) == ["model_directory", "version"]


# --------------------------------------------------------------------------- #
# main(): download_metadata and the outdated flag
# --------------------------------------------------------------------------- #
GEMMA_REPO = ("llamacpp", "google", "gemma-4-E2B-it-qat-q4_0-gguf")

CURRENT_METADATA = """\
models:
- downloaded_at: '2026-08-03T09:41:12Z'
  handler: hf-handler
  model_id: llamacpp:gemma-4-E2B_q4_0-it
  model_origin: built_in
  files:
  - gemma-4-E2B_q4_0-it.gguf
  inputs:
    model_url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf"
    models_repository: llamacpp
    model_directory: google/gemma-4-E2B-it-qat-q4_0-gguf
"""


def _install_gemma(models_dir, metadata_text=None):
    repo = os.path.join(str(models_dir), *GEMMA_REPO)
    _make_gguf(os.path.join(repo, "gemma-4-E2B_q4_0-it.gguf"))
    if metadata_text is not None:
        _write_metadata_file(repo, metadata_text)
    return repo


def _gemma_entry(models):
    entries = [m for m in models if m["id"] == "llamacpp:gemma-4-E2B_q4_0-it"]
    assert len(entries) == 1
    return entries[0]


def test_main_surfaces_download_metadata(monkeypatch, capsys, tmp_path):
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir, CURRENT_METADATA)

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is True
    assert entry["download_metadata"]["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert entry["download_metadata"]["inputs"]["models_repository"] == "llamacpp"
    # The declaration is unchanged since the download.
    assert entry["outdated"] is False
    assert "outdated_fields" not in entry


def test_main_flags_outdated_when_inputs_differ(monkeypatch, capsys, tmp_path):
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    # Downloaded from a different (older) revision than the YAML now declares.
    _install_gemma(models_dir, CURRENT_METADATA.replace("/blob/main/", "/blob/0ldsha/"))

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is True
    assert entry["outdated"] is True
    assert entry["outdated_fields"] == ["model_url"]


def test_main_no_metadata_key_when_absent(monkeypatch, capsys, tmp_path):
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir)

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is True
    # A legacy install: unknown, not "up to date".
    assert "download_metadata" not in entry
    assert "outdated" not in entry


def test_main_corrupt_metadata_is_ignored(monkeypatch, capsys, tmp_path):
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir, "a: b: c\n")

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is True
    assert "download_metadata" not in entry


def test_main_never_emits_internal_variables(monkeypatch, capsys, tmp_path):
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    assert all("variables" not in m for m in models)


def test_main_lists_an_unlisted_model_with_its_metadata(monkeypatch, capsys, tmp_path):
    """A Hugging Face model downloaded ad hoc, with no models-list.yaml entry.

    Any repository can be pulled by --model-key / --model-repo-id / --model-url, so
    the listing must surface it from the filesystem, carry its record, and say
    nothing about whether it is outdated (there is no declaration to compare to).
    """
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    repo = os.path.join(str(models_dir), "llamacpp", "TheBloke", "Mistral-7B-Instruct-v0.2-GGUF")
    _make_gguf(os.path.join(repo, "mistral.Q4_0.gguf"))
    _write_metadata_file(
        repo,
        "models:\n- handler: hf-handler\n  model_id: llamacpp:mistral.Q4_0\n  model_origin: user\n  files: [mistral.Q4_0.gguf]\n",
    )

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entries = [m for m in models if m["id"] == "llamacpp:TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral.Q4_0"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["installed"] is True
    assert entry["model_origin"] == "user"
    assert entry["download_metadata"]["model_origin"] == "user"
    assert "outdated" not in entry
    # The models-list.yaml entries are unaffected.
    assert _gemma_entry(models)["installed"] is False


# --------------------------------------------------------------------------- #
# model_origin
# --------------------------------------------------------------------------- #
def test_main_marks_declared_models_as_builtin(monkeypatch, capsys, tmp_path):
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    assert models
    assert all(m["model_origin"] == "built_in" for m in models)


def test_main_keeps_builtin_origin_after_the_filesystem_merge(monkeypatch, capsys, tmp_path):
    """A declared model found on disk stays curated; the merge must not relabel it."""
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir, CURRENT_METADATA)

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is True
    assert entry["model_origin"] == "built_in"


def test_main_pre_loaded_models_are_builtin(monkeypatch, capsys, tmp_path):
    yaml_text = 'models:\n - "builtin:asr":\n    name: "Builtin ASR"\n    deployment:\n      handler: "ai-hub-handler"\n      pre-loaded: true\n'
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, yaml_text)
    assert models[0]["model_origin"] == "built_in"


def test_find_llamacpp_models_are_user(tmp_path):
    base = tmp_path / "models"
    _make_gguf(os.path.join(str(base), "llamacpp", "TheBloke", "Mistral-GGUF", "mistral.Q4_0.gguf"))
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["model_origin"] == "user"


def test_table_shows_the_origin_column(monkeypatch, capsys, tmp_path):
    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text(SAMPLE_YAML)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    monkeypatch.setattr("sys.argv", ["list_models.py", "--models-dir", str(models_dir), "--model-list", str(yaml_path)])
    list_models.main()
    out = capsys.readouterr().out
    assert "ORIGIN" in out
    assert "built_in" in out


def test_find_llamacpp_attaches_metadata(tmp_path):
    base = tmp_path / "models"
    repo = os.path.join(str(base), *GEMMA_REPO)
    _make_gguf(os.path.join(repo, "gemma-4-E2B_q4_0-it.gguf"))
    _write_metadata_file(repo, CURRENT_METADATA)
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["download_metadata"]["handler"] == "hf-handler"


def test_find_llamacpp_attaches_metadata_to_marker_only_entry(tmp_path):
    base = tmp_path / "models"
    repo = base.joinpath(*GEMMA_REPO)
    repo.mkdir(parents=True)
    write_marker(str(repo), handler="hf-handler", model_url="https://hf/repo/gemma-4-E2B_q4_0-it.gguf")
    _write_metadata_file(str(repo), CURRENT_METADATA)
    results = list_models.find_llamacpp_models(str(base))
    assert len(results) == 1
    assert results[0]["downloading"] is True
    assert results[0]["download_metadata"]["handler"] == "hf-handler"


# --------------------------------------------------------------------------- #
# Several quantizations sharing one repository directory, each with its record
# --------------------------------------------------------------------------- #
def _write_quant_record(repo, quantization, files, model_id=None):
    """A record the way hf_downloader writes one: merged, naming its files."""
    write_metadata(
        repo,
        "hf-handler",
        env={"model_url": f"unsloth/Qwen3-0.6B-GGUF:{quantization}", "models_repository": "llamacpp"},
        models_list_path="",
        fallback_model_id=model_id or f"llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-{quantization}",
        files=files,
    )


def test_find_llamacpp_attaches_each_quantizations_own_record(tmp_path):
    """Two quantizations in one repository directory: each entry must carry the
    record of its own download, not whichever download happened last."""
    base = tmp_path / "models"
    repo = os.path.join(str(base), "llamacpp", "unsloth", "Qwen3-0.6B-GGUF")
    _make_gguf(os.path.join(repo, "Qwen3-0.6B-Q4_0.gguf"))
    _make_gguf(os.path.join(repo, "Qwen3-0.6B-Q8_0.gguf"))
    _write_quant_record(repo, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant_record(repo, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])

    results = {entry["id"]: entry for entry in list_models.find_llamacpp_models(str(base))}

    for quantization in ("Q4_0", "Q8_0"):
        entry = results[f"llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-{quantization}"]
        assert entry["download_metadata"]["files"] == [f"Qwen3-0.6B-{quantization}.gguf"]
        assert entry["download_metadata"]["inputs"]["model_url"].endswith(f":{quantization}")


def test_find_llamacpp_does_not_borrow_a_siblings_record(tmp_path):
    """A quantization the records do not claim (installed by an older downloader,
    say) is unknown — reporting a sibling's record would misdescribe it."""
    base = tmp_path / "models"
    repo = os.path.join(str(base), "llamacpp", "unsloth", "Qwen3-0.6B-GGUF")
    _make_gguf(os.path.join(repo, "Qwen3-0.6B-Q4_0.gguf"))
    _make_gguf(os.path.join(repo, "Qwen3-0.6B-Q8_0.gguf"))
    _write_quant_record(repo, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])

    results = {entry["id"]: entry for entry in list_models.find_llamacpp_models(str(base))}

    assert "download_metadata" in results["llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0"]
    # Unclaimed means unrecorded: out-of-the-box treatment — stem name, no record.
    assert "download_metadata" not in results["llamacpp:Qwen3-0.6B-Q8_0"]


def test_model_metadata_picks_the_entrys_own_record(tmp_path):
    """Two curated entries share the repository directory; each reads its own record."""
    base = tmp_path / "models"
    repo = os.path.join(str(base), "llamacpp", "unsloth", "Qwen3-0.6B-GGUF")
    _write_quant_record(repo, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"], model_id="llamacpp:entry-q4")
    _write_quant_record(repo, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"], model_id="llamacpp:entry-q8")
    model_info = {"model_directory": "unsloth/Qwen3-0.6B-GGUF", "models_repository": "llamacpp"}

    q4 = list_models.model_metadata(dict(model_info, id="llamacpp:entry-q4"), str(base))
    assert q4["files"] == ["Qwen3-0.6B-Q4_0.gguf"]
    q8 = list_models.model_metadata(dict(model_info, id="llamacpp:entry-q8"), str(base))
    assert q8["files"] == ["Qwen3-0.6B-Q8_0.gguf"]
    # An entry none of the records name gets no record — never a sibling's, whose
    # inputs would flag this model as outdated against the wrong download.
    assert list_models.model_metadata(dict(model_info, id="llamacpp:entry-q2"), str(base)) is None


def test_main_supported_board_filter(monkeypatch, capsys, tmp_path):
    yaml_text = (
        SAMPLE_YAML
        + """\
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
    )
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, yaml_text, extra_args=["--supported-board", "ventunoq"])
    ids = {m["id"] for m in models}
    assert "ei:unoq-only" not in ids
    assert "llamacpp:gemma-4-E2B_q4_0-it" in ids


def test_main_installed_only_filter(monkeypatch, capsys, tmp_path):
    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text(SAMPLE_YAML)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _make_gguf(str(models_dir / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf" / "gemma-4-E2B_q4_0-it.gguf"))
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


# --------------------------------------------------------------------------- #
# Identical GGUF file names from different repositories
# --------------------------------------------------------------------------- #
def test_find_llamacpp_same_file_name_in_two_repos_lists_two_models(tmp_path):
    """Two repositories publishing the same file name are two models, not one.

    Each gets a path-qualified id with its own path, size and download record —
    the file-stem id would alias them, and delete/status/size would act on
    whichever file happened to win.
    """
    base = tmp_path / "models"
    unsloth = base / "llamacpp" / "unsloth" / "SmolLM2-GGUF"
    bartowski = base / "llamacpp" / "bartowski" / "SmolLM2-GGUF"
    _make_gguf(str(unsloth / "SmolLM2-Q4_K_M.gguf"), size_bytes=1024 * 1024)
    _make_gguf(str(bartowski / "SmolLM2-Q4_K_M.gguf"), size_bytes=2 * 1024 * 1024)
    _write_metadata_file(
        str(unsloth),
        "models:\n- model_origin: user\n  files: [SmolLM2-Q4_K_M.gguf]\n  inputs:\n    model_url: llamacpp:unsloth/SmolLM2-GGUF:Q4_K_M\n",
    )
    _write_metadata_file(
        str(bartowski),
        "models:\n- model_origin: user\n  files: [SmolLM2-Q4_K_M.gguf]\n  inputs:\n    model_url: llamacpp:bartowski/SmolLM2-GGUF:Q4_K_M\n",
    )

    results = {entry["id"]: entry for entry in list_models.find_llamacpp_models(str(base))}

    assert set(results) == {
        "llamacpp:unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M",
        "llamacpp:bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M",
    }
    for owner, size in (("unsloth", 1.0), ("bartowski", 2.0)):
        entry = results[f"llamacpp:{owner}/SmolLM2-GGUF/SmolLM2-Q4_K_M"]
        assert entry["path"].endswith(f"{owner}/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf")
        assert entry["disk_size_mb"] == size
        # Each entry carries its own download record, never the other repository's.
        assert owner in entry["download_metadata"]["inputs"]["model_url"]


def test_main_lists_same_named_ad_hoc_downloads_separately(monkeypatch, capsys, tmp_path):
    """The reported bug: two ad-hoc downloads sharing a file name merged into one entry."""
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _make_gguf(str(models_dir / "llamacpp" / "unsloth" / "SmolLM2-GGUF" / "SmolLM2-Q4_K_M.gguf"))
    _make_gguf(str(models_dir / "llamacpp" / "bartowski" / "SmolLM2-GGUF" / "SmolLM2-Q4_K_M.gguf"))
    _record_install(models_dir / "llamacpp" / "unsloth" / "SmolLM2-GGUF", "user", ["SmolLM2-Q4_K_M.gguf"])
    _record_install(models_dir / "llamacpp" / "bartowski" / "SmolLM2-GGUF", "user", ["SmolLM2-Q4_K_M.gguf"])

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    smol = sorted(m["id"] for m in models if "SmolLM2" in m["id"])
    assert smol == [
        "llamacpp:bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M",
        "llamacpp:unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M",
    ]


def test_main_ad_hoc_download_cannot_masquerade_as_a_declared_model(monkeypatch, capsys, tmp_path):
    """A file named like a curated model, from another repository, is not that model.

    The declared entry stays not-installed and the ad-hoc file is listed under a
    path-qualified id: merging them would let delete/status act on the wrong files.
    """
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _make_gguf(str(models_dir / "llamacpp" / "bartowski" / "gemma-clone-GGUF" / "gemma-4-E2B_q4_0-it.gguf"))

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    assert _gemma_entry(models)["installed"] is False
    clones = [m for m in models if m["id"] == "llamacpp:bartowski/gemma-clone-GGUF/gemma-4-E2B_q4_0-it"]
    assert len(clones) == 1
    assert clones[0]["installed"] is True
    assert clones[0]["model_origin"] == "user"


def test_main_declared_model_merges_next_to_its_same_named_clone(monkeypatch, capsys, tmp_path):
    """With both the declared file and a same-named clone on disk, each keeps its identity."""
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir)
    _make_gguf(str(models_dir / "llamacpp" / "bartowski" / "gemma-clone-GGUF" / "gemma-4-E2B_q4_0-it.gguf"))

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    gemma = _gemma_entry(models)
    assert gemma["installed"] is True
    assert gemma["path"].endswith("google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf")
    clones = [m for m in models if m["id"] == "llamacpp:bartowski/gemma-clone-GGUF/gemma-4-E2B_q4_0-it"]
    assert len(clones) == 1


def test_main_merges_a_pending_declared_download_by_location(monkeypatch, capsys, tmp_path):
    """A curated download in progress (marker, no GGUF yet) folds into its entry."""
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    repo = os.path.join(str(models_dir), *GEMMA_REPO)
    os.makedirs(repo)
    write_marker(
        str(repo),
        handler="hf-handler",
        model_url="https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf",
    )

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    entry = _gemma_entry(models)
    assert entry["installed"] is False
    assert entry["downloading"] is True
    assert entry["model_origin"] == "built_in"


def test_main_never_emits_location_keys(monkeypatch, capsys, tmp_path):
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    _install_gemma(models_dir)
    _make_gguf(str(models_dir / "llamacpp" / "TheBloke" / "Mistral-GGUF" / "mistral.Q4_0.gguf"))

    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    assert all(not key.startswith("_") for m in models for key in m)


def test_main_supports_promoting_an_ad_hoc_model_to_the_curated_list(monkeypatch, capsys, tmp_path):
    """An ad-hoc install adopted by a later catalog release becomes that curated model.

    Nothing on the board changes — the same files and record are re-read against the
    new catalog: the entry lists as installed under its curated id (the old path
    id disappears), and the recorded inputs, made against no declaration, flag it
    outdated so the host can offer the pinned re-download.
    """
    promoted_yaml = (
        SAMPLE_YAML
        + """\
 - "llamacpp:SmolLM2-135M-Instruct-Q4_K_M":
    name: "SmolLM2 135M"
    supported_boards: ["ventunoq"]
    deployment:
      handler: "hf-handler"
      platforms:
        - ventunoq:
            variables:
              model_url: "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/blob/pinned0sha/SmolLM2-135M-Instruct-Q4_K_M.gguf"
              models_repository: "llamacpp"
              model_directory: "unsloth/SmolLM2-135M-Instruct-GGUF"
"""
    )
    # Release N: downloaded ad hoc, no catalog entry.
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    repo = models_dir / "llamacpp" / "unsloth" / "SmolLM2-135M-Instruct-GGUF"
    _make_gguf(str(repo / "SmolLM2-135M-Instruct-Q4_K_M.gguf"))
    _write_metadata_file(
        str(repo),
        "models:\n"
        "- model_origin: user\n"
        "  model_id: llamacpp:unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M\n"
        "  files: [SmolLM2-135M-Instruct-Q4_K_M.gguf]\n"
        "  inputs:\n"
        "    model_url: llamacpp:unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M\n"
        "    models_repository: llamacpp\n"
        "    model_directory: unsloth/SmolLM2-135M-Instruct-GGUF\n",
    )
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, SAMPLE_YAML)
    assert [m["id"] for m in models if "SmolLM2" in m["id"]] == ["llamacpp:unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M"]

    # Release N+1: the catalog now declares the same location.
    list_models._SEARCH_DIR_CACHE.clear()
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, promoted_yaml)
    smol = [m for m in models if "SmolLM2" in m["id"]]
    assert len(smol) == 1
    entry = smol[0]
    assert entry["id"] == "llamacpp:SmolLM2-135M-Instruct-Q4_K_M"
    assert entry["model_origin"] == "built_in"
    assert entry["installed"] is True
    # Downloaded with the ad-hoc inputs, not the pinned URL the entry declares.
    assert entry["outdated"] is True
    assert entry["outdated_fields"] == ["model_url"]


MULTI_BOARD_YAML = """\
models:
 - "llamacpp:multi-Q4_0":
    name: "Multi board"
    supported_boards: ["ventunoq", "unoq"]
    deployment:
      handler: "hf-handler"
      platforms:
        - ventunoq:
            variables:
              model_url: "https://huggingface.co/org/multi-gguf/blob/ventunosha/multi-Q4_0.gguf"
              models_repository: "llamacpp"
              model_directory: "org/multi-gguf"
        - unoq:
            variables:
              model_url: "https://huggingface.co/org/multi-gguf/blob/unosha/multi-Q4_0.gguf"
              models_repository: "llamacpp"
              model_directory: "org/multi-gguf"
"""


def test_main_lists_a_multi_platform_entry_once(monkeypatch, capsys, tmp_path):
    """An entry declaring several board platforms is one model, not one row per board."""
    models_dir, models = _run_main(monkeypatch, capsys, tmp_path, MULTI_BOARD_YAML)
    assert [m["id"] for m in models] == ["llamacpp:multi-Q4_0"]

    # And the single row is the one the filesystem merge updates.
    _make_gguf(str(models_dir / "llamacpp" / "org" / "multi-gguf" / "multi-Q4_0.gguf"))
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, MULTI_BOARD_YAML)
    assert len(models) == 1
    assert models[0]["installed"] is True


def test_main_dedup_prefers_the_platform_of_the_listed_board(monkeypatch, capsys, tmp_path):
    """When per-board variables differ, the outdated check must use this board's."""
    models_dir, _models = _run_main(monkeypatch, capsys, tmp_path, MULTI_BOARD_YAML)
    repo = models_dir / "llamacpp" / "org" / "multi-gguf"
    _make_gguf(str(repo / "multi-Q4_0.gguf"))
    # Downloaded from the revision the unoq platform declares.
    _write_metadata_file(
        str(repo),
        "models:\n"
        "- model_id: llamacpp:multi-Q4_0\n"
        "  files: [multi-Q4_0.gguf]\n"
        "  inputs:\n"
        '    model_url: "https://huggingface.co/org/multi-gguf/blob/unosha/multi-Q4_0.gguf"\n'
        "    models_repository: llamacpp\n"
        "    model_directory: org/multi-gguf\n",
    )

    monkeypatch.setenv("BOARD_NAME", "unoq")
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, MULTI_BOARD_YAML)
    assert models[0]["outdated"] is False

    monkeypatch.setenv("BOARD_NAME", "ventunoq")
    list_models._SEARCH_DIR_CACHE.clear()
    _models_dir, models = _run_main(monkeypatch, capsys, tmp_path, MULTI_BOARD_YAML)
    assert models[0]["outdated"] is True
    assert models[0]["outdated_fields"] == ["model_url"]


def test_gguf_basename():
    url = "https://huggingface.co/org/repo/blob/main/model-Q4_0.gguf"
    assert list_models.gguf_basename(url) == "model-Q4_0.gguf"
    assert list_models.gguf_basename(url + "?download=true") == "model-Q4_0.gguf"
    # A compact key pins a file only when its quantization field names one.
    assert list_models.gguf_basename("llamacpp:org/repo:model-Q4_0.gguf") == "model-Q4_0.gguf"
    assert list_models.gguf_basename("llamacpp:org/repo:Q4_K_M") is None
    assert list_models.gguf_basename("org/repo") is None
    assert list_models.gguf_basename("") is None


def test_main_missing_yaml_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["list_models.py", "--model-list", str(tmp_path / "nope.yaml"), "--json"],
    )
    with pytest.raises(SystemExit) as exc:
        list_models.main()
    assert exc.value.code == 1
