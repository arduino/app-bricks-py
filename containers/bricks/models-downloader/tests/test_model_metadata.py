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
    ORIGIN_USER_CONFIGURED,
    collect_inputs,
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
    """Only the id is taken from the entry; its name/source/... are not duplicated."""
    identity = identify_model(AI_HUB_ENV, _models_list(tmp_path))
    assert identity == {"model_id": "genie:qwen3_4b_instruct_2507", "model_origin": "builtin"}


def test_identify_model_prefers_model_id_env(tmp_path):
    env = dict(AI_HUB_ENV, model_id="host:provided")
    identity = identify_model(env, _models_list(tmp_path))
    assert identity == {"model_id": "host:provided", "model_origin": "builtin"}


def test_identify_model_user_configured_when_yaml_missing(tmp_path):
    identity = identify_model(AI_HUB_ENV, str(tmp_path / "nope.yaml"))
    assert identity == {"model_id": None, "model_origin": "user_configured"}


def test_identify_model_user_configured_when_yaml_is_broken(tmp_path):
    broken = tmp_path / "models-list.yaml"
    broken.write_text("models: [unclosed\n")
    assert identify_model(AI_HUB_ENV, str(broken))["model_origin"] == "user_configured"


def test_identify_model_user_configured_when_no_entry_matches(tmp_path):
    identity = identify_model({"model_name": "something-else"}, _models_list(tmp_path))
    assert identity == {"model_id": None, "model_origin": "user_configured"}


def test_identify_model_uses_the_fallback_id_when_not_curated(tmp_path):
    """An ad-hoc download still needs an id: nothing can refer to a null one."""
    env = {"model_url": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q4_0", "models_repository": "llamacpp"}
    identity = identify_model(env, _models_list(tmp_path), fallback_model_id="llamacpp:mistral.Q4_0")
    assert identity == {"model_id": "llamacpp:mistral.Q4_0", "model_origin": "user_configured"}


def test_identify_model_ignores_the_fallback_when_curated(tmp_path):
    """A declared model keeps its entry key; the fallback is only for the other case."""
    identity = identify_model(AI_HUB_ENV, _models_list(tmp_path), fallback_model_id="should:not:be:used")
    assert identity == {"model_id": "genie:qwen3_4b_instruct_2507", "model_origin": "builtin"}


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
# metadata_payload
# --------------------------------------------------------------------------- #
def test_metadata_payload_drops_empty_values():
    payload = metadata_payload(
        "hf-handler",
        inputs={"model_name": "x", "quantization": "", "version": None},
        identity={"model_id": "a:b", "model_origin": "builtin"},
    )
    assert payload["inputs"] == {"model_name": "x"}
    assert payload["model_id"] == "a:b"


def test_metadata_payload_omits_empty_inputs():
    payload = metadata_payload("ei-handler")
    assert list(payload) == ["downloaded_at", "handler", "model_id", "model_origin"]
    assert payload["model_id"] is None
    assert payload["model_origin"] == "user_configured"


def test_metadata_payload_copies_nothing_else_from_the_entry():
    """Only the id points back at models-list.yaml; its other fields are not duplicated."""
    payload = metadata_payload(
        "ai-hub-handler",
        inputs={"model_name": "x"},
        identity={"model_id": "genie:x", "model_origin": "builtin"},
    )
    assert list(payload) == ["downloaded_at", "handler", "model_id", "model_origin", "inputs"]
    for copied in ("name", "source", "source_model_id", "source_model_url", "model_size_mb", "resolved"):
        assert copied not in payload


# --------------------------------------------------------------------------- #
# write_metadata / read_metadata
# --------------------------------------------------------------------------- #
def test_write_read_roundtrip(tmp_path):
    path = write_metadata(str(tmp_path), "ai-hub-handler", env=AI_HUB_ENV, models_list_path="")
    assert path == str(tmp_path / METADATA_NAME)
    # Exactly one file: the atomic ".tmp" sibling must be gone.
    assert os.listdir(tmp_path) == [METADATA_NAME]
    data = read_metadata(str(tmp_path))
    assert data["handler"] == "ai-hub-handler"
    assert data["inputs"] == AI_HUB_ENV


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
    write_metadata(str(tmp_path), "hf-handler", env=dict(HF_ENV, model_directory="old/repo"), models_list_path="")
    write_metadata(str(tmp_path), "hf-handler", env=dict(HF_ENV, model_directory="new/repo"), models_list_path="")
    assert os.listdir(tmp_path) == [METADATA_NAME]
    assert read_metadata(str(tmp_path))["inputs"]["model_directory"] == "new/repo"


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


def test_read_keeps_unknown_keys(tmp_path):
    """A file written by a newer version must not break an older reader."""
    (tmp_path / METADATA_NAME).write_text("handler: future-handler\nsomething_new: 42\n")
    data = read_metadata(str(tmp_path))
    assert data["handler"] == "future-handler"
    assert data["something_new"] == 42


# --------------------------------------------------------------------------- #
# End-to-end payload shape, per handler
# --------------------------------------------------------------------------- #
PAYLOAD_KEYS = ["downloaded_at", "handler", "model_id", "model_origin", "inputs"]


def test_payload_ai_hub(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_metadata(str(model_dir), "ai-hub-handler", env=AI_HUB_ENV, models_list_path=_models_list(tmp_path))
    data = read_metadata(str(model_dir))
    assert list(data) == PAYLOAD_KEYS
    assert data["handler"] == "ai-hub-handler"
    assert data["model_id"] == "genie:qwen3_4b_instruct_2507"
    assert data["model_origin"] == "builtin"
    assert data["inputs"] == AI_HUB_ENV


def test_payload_edge_impulse(tmp_path):
    model_dir = tmp_path / "efficientnet-b4-qnn"
    model_dir.mkdir()
    write_metadata(str(model_dir), "ei-handler", env=EI_ENV, models_list_path=_models_list(tmp_path))
    data = read_metadata(str(model_dir))
    assert list(data) == PAYLOAD_KEYS
    assert data["model_id"] == "ei:efficientnet-b4"
    assert data["inputs"] == EI_ENV
    # YAML ints arrive as strings and are recorded verbatim.
    assert data["inputs"]["ei_project_id"] == "948887"
    # quantization is not set for this model: the key is omitted, not written empty.
    assert "quantization" not in data["inputs"]


def test_payload_hugging_face(tmp_path):
    model_dir = tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    model_dir.mkdir(parents=True)
    write_metadata(str(model_dir), "hf-handler", env=HF_ENV, models_list_path=_models_list(tmp_path))
    data = read_metadata(str(model_dir))
    assert list(data) == PAYLOAD_KEYS
    assert data["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert data["inputs"] == HF_ENV
    # The pinned commit is already in the recorded model_url.
    assert "1894d1fc" in data["inputs"]["model_url"]


def test_payload_for_a_repo_absent_from_models_list(tmp_path):
    """An ad-hoc Hugging Face download is fully supported and fully identified.

    Any repository can be pulled by putting its URL or compact key in model_url, with
    no models-list.yaml entry. The record must still be written, name the model, and
    say the model is user-configured.
    """
    model_dir = tmp_path / "TheBloke" / "Mistral-7B-Instruct-v0.2-GGUF"
    model_dir.mkdir(parents=True)
    env = {
        # model_url in its compact-key syntax.
        "model_url": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q4_0",
        "models_repository": "llamacpp",
        "model_directory": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    }
    path = write_metadata(
        str(model_dir),
        "hf-handler",
        env=env,
        models_list_path=_models_list(tmp_path),
        fallback_model_id="llamacpp:mistral-7b-instruct-v0.2.Q4_0",
    )
    assert path is not None
    data = read_metadata(str(model_dir))
    assert list(data) == PAYLOAD_KEYS
    assert data["model_id"] == "llamacpp:mistral-7b-instruct-v0.2.Q4_0"
    assert data["model_origin"] == "user_configured"
    # The download is still fully described by its own variables.
    assert data["inputs"] == env


def test_written_file_is_plain_safe_yaml(tmp_path):
    """No python-specific tags: the file must be readable by any YAML parser."""
    write_metadata(str(tmp_path), "hf-handler", env=HF_ENV, models_list_path="")
    text = (tmp_path / METADATA_NAME).read_text()
    assert "!!python" not in text
    assert yaml.safe_load(text)["inputs"] == HF_ENV


def test_write_uses_a_passed_identity(tmp_path):
    """A handler reporting the id to the host records the same one, without re-resolving."""
    identity = {"model_id": "llamacpp:from-caller", "model_origin": ORIGIN_USER_CONFIGURED}
    write_metadata(str(tmp_path), "hf-handler", env=HF_ENV, models_list_path="", identity=identity)

    data = read_metadata(str(tmp_path))
    assert data["model_id"] == "llamacpp:from-caller"
    assert data["model_origin"] == ORIGIN_USER_CONFIGURED


def test_write_resolves_the_identity_when_none_is_passed(tmp_path):
    write_metadata(str(tmp_path), "hf-handler", env=HF_ENV, models_list_path="", fallback_model_id="llamacpp:derived")
    assert read_metadata(str(tmp_path))["model_id"] == "llamacpp:derived"
