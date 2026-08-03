# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``common/model_metadata.py``."""

import json
import os
import re

import yaml

from common.model_metadata import (
    METADATA_NAME,
    collect_inputs,
    dir_stats,
    identify_model,
    is_bookkeeping_name,
    metadata_payload,
    read_metadata,
    utc_now_iso,
    write_metadata,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
MODELS_LIST = """\
models:
 - "genie:qwen3_4b_instruct_2507":
    name: "Qwen 3-4B Instruct"
    deployment:
      handler: "ai-hub-handler"
      platforms:
        - ventunoq:
            variables:
              model_type: "genie"
              model_name: "qwen3_4b_instruct_2507"
              models_repository: "genai"
              model_directory: "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs8275"
              quantization: "w4a16"
              chipset: "qualcomm-qcs8275"
              version: "0.51.0"
    metadata:
      model_size_mb: 3039
      source: "qualcomm-ai-hub"
      source-model-id: "qwen3_4b_instruct_2507"
      source-model-url: "https://aihub.qualcomm.com/models/qwen3_4b_instruct_2507"
 - "ei:efficientnet-b4":
    name: "EfficientNet-B4"
    deployment:
      handler: "ei-handler"
      platforms:
        - ventunoq:
            variables:
              ei_project_id: 948887
              ei_impulse_id: 10
              models_repository: "edge-impulse"
              model_name: "efficientnet-b4-qnn.eim"
              target: "runner-linux-aarch64-qnn"
    metadata:
      source: "edgeimpulse"
      source-model-id: "efficientnet_b4"
 - "llamacpp:gemma-4-E2B_q4_0-it":
    name: "Gemma 4 E2B"
    deployment:
      handler: "hf-handler"
      platforms:
        - ventunoq:
            variables:
              model_url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/1894d1fc/gemma-4-E2B_q4_0-it.gguf"
              models_repository: "llamacpp"
              model_directory: "google/gemma-4-E2B-it-qat-q4_0-gguf"
    metadata:
      source: "huggingface"
      source-model-url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf"
"""

AI_HUB_ENV = {
    "model_type": "genie",
    "model_name": "qwen3_4b_instruct_2507",
    "models_repository": "genai",
    "model_directory": "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs8275",
    "quantization": "w4a16",
    "chipset": "qualcomm-qcs8275",
    "version": "0.51.0",
}

EI_ENV = {
    "ei_project_id": "948887",
    "ei_impulse_id": "10",
    "models_repository": "edge-impulse",
    "model_name": "efficientnet-b4-qnn.eim",
    "target": "runner-linux-aarch64-qnn",
}

HF_ENV = {
    "model_url": "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/1894d1fc/gemma-4-E2B_q4_0-it.gguf",
    "models_repository": "llamacpp",
    "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf",
}


def _models_list(tmp_path):
    path = tmp_path / "models-list.yaml"
    path.write_text(MODELS_LIST)
    return str(path)


# --------------------------------------------------------------------------- #
# utc_now_iso / is_bookkeeping_name
# --------------------------------------------------------------------------- #
def test_utc_now_iso_is_second_resolution_zulu():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now_iso())


def test_is_bookkeeping_name():
    assert is_bookkeeping_name(".download")
    assert is_bookkeeping_name(METADATA_NAME)
    # The temporary sibling of an interrupted atomic write must count too.
    assert is_bookkeeping_name(METADATA_NAME + ".tmp")
    assert not is_bookkeeping_name("model.gguf")
    assert not is_bookkeeping_name("models.ini")


# --------------------------------------------------------------------------- #
# collect_inputs
# --------------------------------------------------------------------------- #
def test_collect_inputs_keeps_only_known_non_empty_keys():
    env = dict(AI_HUB_ENV, PATH="/usr/bin", HOME="/home/arduino", BOARD_NAME="ventunoq", model_url="")
    inputs = collect_inputs(env)
    assert inputs == AI_HUB_ENV
    for unexpected in ("PATH", "HOME", "BOARD_NAME", "model_url"):
        assert unexpected not in inputs


def test_collect_inputs_orders_keys_canonically():
    inputs = collect_inputs(AI_HUB_ENV)
    assert list(inputs) == ["models_repository", "model_directory", "model_name", "model_type", "quantization", "chipset", "version"]


def test_collect_inputs_never_records_secrets():
    env = dict(HF_ENV, hf_token="hf_secret", HF_TOKEN="hf_secret", HF_HUB_TOKEN="hf_secret")
    inputs = collect_inputs(env, extra_keys=("hf_token", "HF_TOKEN"))
    assert "hf_token" not in inputs
    assert "HF_TOKEN" not in inputs
    assert "HF_HUB_TOKEN" not in inputs


def test_collect_inputs_extra_keys():
    inputs = collect_inputs(dict(EI_ENV, custom_flag="on"), extra_keys=("custom_flag",))
    assert inputs["custom_flag"] == "on"


def test_collect_inputs_empty_environment():
    assert collect_inputs({}) == {}


# --------------------------------------------------------------------------- #
# identify_model
# --------------------------------------------------------------------------- #
def test_identify_model_from_models_list(tmp_path):
    identity = identify_model(AI_HUB_ENV, _models_list(tmp_path))
    assert identity == {
        "model_id": "genie:qwen3_4b_instruct_2507",
        "model_id_source": "models-list",
        "name": "Qwen 3-4B Instruct",
        "source": "qualcomm-ai-hub",
        "source_model_id": "qwen3_4b_instruct_2507",
        "source_model_url": "https://aihub.qualcomm.com/models/qwen3_4b_instruct_2507",
    }


def test_identify_model_prefers_model_id_env(tmp_path):
    env = dict(AI_HUB_ENV, model_id="host:provided")
    identity = identify_model(env, _models_list(tmp_path))
    assert identity == {"model_id": "host:provided", "model_id_source": "env"}


def test_identify_model_unresolved_when_yaml_missing(tmp_path):
    identity = identify_model(AI_HUB_ENV, str(tmp_path / "nope.yaml"))
    assert identity == {"model_id": None, "model_id_source": "unresolved"}


def test_identify_model_unresolved_when_yaml_is_broken(tmp_path):
    broken = tmp_path / "models-list.yaml"
    broken.write_text("models: [unclosed\n")
    assert identify_model(AI_HUB_ENV, str(broken))["model_id_source"] == "unresolved"


def test_identify_model_unresolved_when_no_entry_matches(tmp_path):
    identity = identify_model({"model_name": "something-else"}, _models_list(tmp_path))
    assert identity == {"model_id": None, "model_id_source": "unresolved"}


def test_identify_model_needs_the_derived_model_directory(tmp_path):
    """models_list.yaml declares model_directory, so an env missing it cannot match.

    The Hugging Face handler derives it from the repo id (a substring of the model
    URL) before writing the record — this is what that derivation buys.
    """
    models_list = _models_list(tmp_path)
    without = {key: value for key, value in HF_ENV.items() if key != "model_directory"}
    assert identify_model(without, models_list)["model_id"] is None

    derived = {**without, "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf"}
    assert identify_model(derived, models_list)["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"


def test_inputs_record_the_derived_model_directory(tmp_path):
    without = {key: value for key, value in HF_ENV.items() if key != "model_directory"}
    write_metadata(
        str(tmp_path),
        "hf-handler",
        env={**without, "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf"},
        models_list_path=_models_list(tmp_path),
    )
    data = read_metadata(str(tmp_path))
    assert data["inputs"]["model_directory"] == "google/gemma-4-E2B-it-qat-q4_0-gguf"
    assert data["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"


# --------------------------------------------------------------------------- #
# dir_stats
# --------------------------------------------------------------------------- #
def test_dir_stats_excludes_marker_metadata_and_cache(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"\0" * 100)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "weights.bin").write_bytes(b"\0" * 50)
    (tmp_path / ".download").write_text("{}")
    (tmp_path / METADATA_NAME).write_text("schema_version: 1\n")
    (tmp_path / (METADATA_NAME + ".tmp")).write_text("partial")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "blob").write_bytes(b"\0" * 999)
    assert dir_stats(str(tmp_path)) == {"file_count": 2, "size_bytes": 150}


def test_dir_stats_missing_dir():
    assert dir_stats("/definitely/not/here") == {"file_count": 0, "size_bytes": 0}


# --------------------------------------------------------------------------- #
# metadata_payload
# --------------------------------------------------------------------------- #
def test_metadata_payload_drops_empty_values():
    payload = metadata_payload(
        "hf-handler",
        inputs={"model_name": "x", "quantization": ""},
        resolved={"revision": None, "file_count": 0, "extracted": False},
        identity={"model_id": "a:b", "model_id_source": "models-list", "name": "N", "source": None},
    )
    assert payload["inputs"] == {"model_name": "x"}
    # 0 and False are real values and must survive; None and "" must not.
    assert payload["resolved"] == {"file_count": 0, "extracted": False}
    assert "source" not in payload
    assert payload["name"] == "N"


def test_metadata_payload_omits_empty_blocks():
    payload = metadata_payload("ei-handler")
    assert list(payload) == ["schema_version", "downloaded_at", "handler", "model_id", "model_id_source"]
    assert payload["model_id"] is None
    assert payload["model_id_source"] == "unresolved"


# --------------------------------------------------------------------------- #
# write_metadata / read_metadata
# --------------------------------------------------------------------------- #
def test_write_read_roundtrip(tmp_path):
    path = write_metadata(str(tmp_path), "ai-hub-handler", resolved={"download_url": "https://x/y.zip"}, env=AI_HUB_ENV, models_list_path="")
    assert path == str(tmp_path / METADATA_NAME)
    # Exactly one file: the atomic ".tmp" sibling must be gone.
    assert os.listdir(tmp_path) == [METADATA_NAME]
    data = read_metadata(str(tmp_path))
    assert data["schema_version"] == 1
    assert data["handler"] == "ai-hub-handler"
    assert data["inputs"] == AI_HUB_ENV
    assert data["resolved"] == {"download_url": "https://x/y.zip"}


def test_write_starts_with_comment_header(tmp_path):
    write_metadata(str(tmp_path), "ai-hub-handler", env={}, models_list_path="")
    text = (tmp_path / METADATA_NAME).read_text()
    assert text.startswith("# Written by the Arduino models-downloader")


def test_write_creates_missing_dir(tmp_path):
    model_dir = tmp_path / "genai" / "some-model"
    path = write_metadata(str(model_dir), "ai-hub-handler", env=AI_HUB_ENV, models_list_path="")
    assert path is not None
    assert os.path.isfile(path)


def test_write_overwrites_previous_record(tmp_path):
    write_metadata(str(tmp_path), "hf-handler", resolved={"revision": "old"}, env=HF_ENV, models_list_path="")
    write_metadata(str(tmp_path), "hf-handler", resolved={"revision": "new"}, env=HF_ENV, models_list_path="")
    assert os.listdir(tmp_path) == [METADATA_NAME]
    assert read_metadata(str(tmp_path))["resolved"]["revision"] == "new"


def test_write_returns_none_and_does_not_raise_on_failure(monkeypatch, capsys, tmp_path):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("common.model_metadata.yaml.safe_dump", _boom)
    assert write_metadata(str(tmp_path), "hf-handler", env=HF_ENV, models_list_path="") is None
    # No leftovers, and the failure is reported as "info" so the host does not
    # mark the completed download as failed.
    assert os.listdir(tmp_path) == []
    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "info"
    assert METADATA_NAME in event["description"]


def test_read_missing_returns_none(tmp_path):
    assert read_metadata(str(tmp_path / METADATA_NAME)) is None
    assert read_metadata(str(tmp_path)) is None


def test_read_corrupt_yaml_returns_none(tmp_path):
    (tmp_path / METADATA_NAME).write_text("a: b: c\n")
    assert read_metadata(str(tmp_path)) is None


def test_read_non_mapping_returns_none(tmp_path):
    (tmp_path / METADATA_NAME).write_text("- a\n- b\n")
    assert read_metadata(str(tmp_path)) is None


def test_read_empty_file_returns_none(tmp_path):
    (tmp_path / METADATA_NAME).write_text("")
    assert read_metadata(str(tmp_path)) is None


def test_read_accepts_dir_or_file_path(tmp_path):
    write_metadata(str(tmp_path), "ei-handler", env=EI_ENV, models_list_path="")
    assert read_metadata(str(tmp_path)) == read_metadata(str(tmp_path / METADATA_NAME))


def test_read_ignores_unknown_schema_version(tmp_path):
    (tmp_path / METADATA_NAME).write_text("schema_version: 99\nhandler: future-handler\n")
    assert read_metadata(str(tmp_path))["handler"] == "future-handler"


# --------------------------------------------------------------------------- #
# End-to-end payload shape, per handler
# --------------------------------------------------------------------------- #
def test_payload_ai_hub(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"\0" * 10)
    write_metadata(
        str(model_dir),
        "ai-hub-handler",
        resolved={
            "download_url": "https://qaihub/x.zip",
            "fetch_command": "qai_hub_models fetch qwen3_4b_instruct_2507 -r genie -p w4a16 -c qualcomm-qcs8275 -v 0.51.0 --url-only",
            "qai_hub_models_version": "0.59.0",
            "extracted": True,
            **dir_stats(str(model_dir)),
        },
        env=AI_HUB_ENV,
        models_list_path=_models_list(tmp_path),
    )
    data = read_metadata(str(model_dir))
    assert list(data) == [
        "schema_version",
        "downloaded_at",
        "handler",
        "model_id",
        "model_id_source",
        "name",
        "source",
        "source_model_id",
        "source_model_url",
        "inputs",
        "resolved",
    ]
    assert data["handler"] == "ai-hub-handler"
    assert data["model_id"] == "genie:qwen3_4b_instruct_2507"
    assert data["model_id_source"] == "models-list"
    assert data["inputs"] == AI_HUB_ENV
    assert data["resolved"]["extracted"] is True
    assert data["resolved"]["file_count"] == 1
    assert data["resolved"]["size_bytes"] == 10


def test_payload_edge_impulse(tmp_path):
    model_dir = tmp_path / "efficientnet-b4-qnn"
    model_dir.mkdir()
    (model_dir / "efficientnet-b4-qnn.eim").write_bytes(b"\0" * 20)
    write_metadata(
        str(model_dir),
        "ei-handler",
        resolved={
            "download_url": "https://studio.edgeimpulse.com/v1/api/948887/deployment/download?type=runner-linux-aarch64-qnn&impulseId=10",
            "ei_project_id": 948887,
            "ei_impulse_id": 10,
            "target": "runner-linux-aarch64-qnn",
            "quantization": None,
            "files": ["efficientnet-b4-qnn.eim"],
            **dir_stats(str(model_dir)),
        },
        env=EI_ENV,
        models_list_path=_models_list(tmp_path),
    )
    data = read_metadata(str(model_dir))
    assert data["model_id"] == "ei:efficientnet-b4"
    assert data["source"] == "edgeimpulse"
    assert "source_model_url" not in data  # absent from the entry's metadata
    assert data["inputs"] == EI_ENV
    # "resolved" holds the parsed ints actually sent, "inputs" the raw environment.
    assert data["resolved"]["ei_project_id"] == 948887
    assert data["inputs"]["ei_project_id"] == "948887"
    assert "quantization" not in data["resolved"]
    assert data["resolved"]["files"] == ["efficientnet-b4-qnn.eim"]


def test_payload_hugging_face(tmp_path):
    model_dir = tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    model_dir.mkdir(parents=True)
    (model_dir / "gemma-4-E2B_q4_0-it.gguf").write_bytes(b"\0" * 30)
    write_metadata(
        str(model_dir),
        "hf-handler",
        resolved={
            "repo_id": "google/gemma-4-E2B-it-qat-q4_0-gguf",
            "revision": "1894d1fc",
            "revision_source": "url",
            "allow_pattern": "gemma-4-E2B_q4_0-it.gguf",
            "mmproj_allow_pattern": None,
            "files": ["gemma-4-E2B_q4_0-it.gguf"],
            **dir_stats(str(model_dir)),
        },
        env=HF_ENV,
        models_list_path=_models_list(tmp_path),
    )
    data = read_metadata(str(model_dir))
    assert data["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert data["name"] == "Gemma 4 E2B"
    assert data["inputs"] == HF_ENV
    assert data["resolved"]["revision"] == "1894d1fc"
    assert data["resolved"]["revision_source"] == "url"
    assert "mmproj_allow_pattern" not in data["resolved"]


def test_payload_for_a_repo_absent_from_models_list(tmp_path):
    """An ad-hoc Hugging Face download is fully supported, just unidentified.

    Any repository can be pulled with --model-key / --model-repo-id / --model-url
    without a models-list.yaml entry, so the record must still be written: the
    identity is reported as unresolved and everything else is intact.
    """
    model_dir = tmp_path / "TheBloke" / "Mistral-7B-Instruct-v0.2-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "mistral.Q4_0.gguf").write_bytes(b"\0" * 40)
    env = {
        "model_key": "llamacpp:TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q4_0",
        "models_repository": "llamacpp",
        "model_directory": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    }
    path = write_metadata(
        str(model_dir),
        "hf-handler",
        resolved={"repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "revision": "abc123", "revision_source": "api", **dir_stats(str(model_dir))},
        env=env,
        models_list_path=_models_list(tmp_path),
    )
    assert path is not None
    data = read_metadata(str(model_dir))
    assert data["model_id"] is None
    assert data["model_id_source"] == "unresolved"
    # No entry to borrow a name or source from.
    assert "name" not in data
    assert "source" not in data
    # The download is still fully described.
    assert data["inputs"] == env
    assert data["resolved"]["repo_id"] == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    assert data["resolved"]["size_bytes"] == 40


def test_written_file_is_plain_safe_yaml(tmp_path):
    """No python-specific tags: the file must be readable by any YAML parser."""
    write_metadata(str(tmp_path), "hf-handler", resolved={"files": ["a.gguf", "b.gguf"]}, env=HF_ENV, models_list_path="")
    text = (tmp_path / METADATA_NAME).read_text()
    assert "!!python" not in text
    assert yaml.safe_load(text)["resolved"]["files"] == ["a.gguf", "b.gguf"]
