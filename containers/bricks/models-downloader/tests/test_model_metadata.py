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
    identify_model,
    is_bookkeeping_name,
    metadata_payload,
    metadata_records,
    prune_metadata_records,
    read_metadata,
    record_for_file,
    record_for_model_id,
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
    assert identity == {"model_id": "genie:qwen3_4b_instruct_2507", "model_origin": "built_in"}


def test_identify_model_prefers_model_id_env(tmp_path):
    env = dict(AI_HUB_ENV, model_id="host:provided")
    identity = identify_model(env, _models_list(tmp_path))
    assert identity == {"model_id": "host:provided", "model_origin": "built_in"}


def test_identify_model_user_when_yaml_missing(tmp_path):
    identity = identify_model(AI_HUB_ENV, str(tmp_path / "nope.yaml"))
    assert identity == {"model_id": None, "model_origin": "user"}


def test_identify_model_user_when_yaml_is_broken(tmp_path):
    broken = tmp_path / "models-list.yaml"
    broken.write_text("models: [unclosed\n")
    assert identify_model(AI_HUB_ENV, str(broken))["model_origin"] == "user"


def test_identify_model_user_when_no_entry_matches(tmp_path):
    identity = identify_model({"model_name": "something-else"}, _models_list(tmp_path))
    assert identity == {"model_id": None, "model_origin": "user"}


def test_identify_model_uses_the_fallback_id_when_not_curated(tmp_path):
    """An ad-hoc download still needs an id: nothing can refer to a null one."""
    env = {"model_url": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q4_0", "models_repository": "llamacpp"}
    identity = identify_model(env, _models_list(tmp_path), fallback_model_id="llamacpp:mistral.Q4_0")
    assert identity == {"model_id": "llamacpp:mistral.Q4_0", "model_origin": "user"}


def test_identify_model_ignores_the_fallback_when_curated(tmp_path):
    """A declared model keeps its entry key; the fallback is only for the other case."""
    identity = identify_model(AI_HUB_ENV, _models_list(tmp_path), fallback_model_id="should:not:be:used")
    assert identity == {"model_id": "genie:qwen3_4b_instruct_2507", "model_origin": "built_in"}


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
    record = metadata_records(read_metadata(str(tmp_path)))[0]
    assert record["inputs"]["model_directory"] == "google/gemma-4-E2B-it-qat-q4_0-gguf"
    assert record["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"


# --------------------------------------------------------------------------- #
# metadata_payload
# --------------------------------------------------------------------------- #
def test_metadata_payload_drops_empty_values():
    payload = metadata_payload(
        "hf-handler",
        inputs={"model_name": "x", "quantization": "", "version": None},
        identity={"model_id": "a:b", "model_origin": "built_in"},
    )
    assert payload["inputs"] == {"model_name": "x"}
    assert payload["model_id"] == "a:b"


def test_metadata_payload_omits_empty_inputs():
    payload = metadata_payload("ei-handler")
    assert list(payload) == ["downloaded_at", "handler", "model_id", "model_origin"]
    assert payload["model_id"] is None
    assert payload["model_origin"] == "user"


def test_metadata_payload_copies_nothing_else_from_the_entry():
    """Only the id points back at models-list.yaml; its other fields are not duplicated."""
    payload = metadata_payload(
        "ai-hub-handler",
        inputs={"model_name": "x"},
        identity={"model_id": "genie:x", "model_origin": "built_in"},
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
    # One document shape for every handler: the records list and nothing else.
    assert list(data) == ["models"]
    record = metadata_records(data)[0]
    assert record["handler"] == "ai-hub-handler"
    assert record["inputs"] == AI_HUB_ENV


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
    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["inputs"]["model_directory"] for r in records] == ["new/repo"]


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
# Multi-record files: several quantizations sharing one repository directory
#
# The Hugging Face handler downloads every quantization of a repository into the
# same directory, so its ".arduino_metadata.yaml" must hold one record per
# installed download instead of letting the last one overwrite the others.
# --------------------------------------------------------------------------- #
def _write_quant(model_dir, quantization, files, model_id=None):
    """A download record the way hf_downloader writes one for *quantization*."""
    return write_metadata(
        str(model_dir),
        "hf-handler",
        env={
            "model_url": f"unsloth/Qwen3-0.6B-GGUF:{quantization}",
            "models_repository": "llamacpp",
            "model_directory": "unsloth/Qwen3-0.6B-GGUF",
        },
        models_list_path="",
        fallback_model_id=model_id or f"llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-{quantization}",
        files=files,
    )


def test_each_download_keeps_its_own_record(tmp_path):
    """The reported bug: downloading Q8_0 must not erase the record of the Q4_0."""
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])

    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["files"] for r in records] == [["Qwen3-0.6B-Q4_0.gguf"], ["Qwen3-0.6B-Q8_0.gguf"]]
    assert [r["inputs"]["model_url"] for r in records] == [
        "unsloth/Qwen3-0.6B-GGUF:Q4_0",
        "unsloth/Qwen3-0.6B-GGUF:Q8_0",
    ]


def test_multi_record_file_holds_nothing_but_the_records(tmp_path):
    """No top-level copy of any record: a single top-level ``model_id`` would misread
    as "the" model of a directory that deliberately holds several."""
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])

    data = read_metadata(str(tmp_path))
    assert list(data) == ["models"]
    assert [r["model_id"] for r in metadata_records(data)] == [
        "llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0",
        "llamacpp:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0",
    ]


def test_redownload_replaces_only_its_own_record(tmp_path):
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])

    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["files"] for r in records] == [["Qwen3-0.6B-Q8_0.gguf"], ["Qwen3-0.6B-Q4_0.gguf"]]


def test_a_new_id_claiming_the_same_file_replaces_the_stale_record(tmp_path):
    """Two records must never both claim the same main file: whichever download
    produced the file last is the one that describes it."""
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"], model_id="llamacpp:old-name")
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"], model_id="llamacpp:new-name")

    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["model_id"] for r in records] == ["llamacpp:new-name"]


def test_a_shared_mmproj_keeps_both_records(tmp_path):
    """Quantizations of a multimodal repository share the mmproj companion, so its
    file appearing in both records must not make one supersede the other."""
    _write_quant(tmp_path, "Q4_0", ["gemma-Q4_0.gguf", "mmproj-F16.gguf"])
    _write_quant(tmp_path, "Q8_0", ["gemma-Q8_0.gguf", "mmproj-F16.gguf"])

    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["files"] for r in records] == [
        ["gemma-Q4_0.gguf", "mmproj-F16.gguf"],
        ["gemma-Q8_0.gguf", "mmproj-F16.gguf"],
    ]


def test_merge_keeps_a_record_that_names_no_files(tmp_path):
    """A record without ``files`` cannot be told apart from any download, so a
    merge never drops it on its own initiative."""
    write_metadata(
        str(tmp_path),
        "hf-handler",
        env={"model_url": "unsloth/Qwen3-0.6B-GGUF:Q4_0", "models_repository": "llamacpp"},
        models_list_path="",
        fallback_model_id="llamacpp:fileless",
    )
    first = metadata_records(read_metadata(str(tmp_path)))[0]

    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])
    records = metadata_records(read_metadata(str(tmp_path)))
    assert records == [first, records[-1]]
    assert records[-1]["files"] == ["Qwen3-0.6B-Q8_0.gguf"]


def test_write_without_files_replaces_the_whole_file(tmp_path):
    """The per-model-directory handlers (AI Hub, Edge Impulse) pass no files: their
    directory holds one model, so the document becomes that one record — in the
    same ``models`` shape every handler writes."""
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    write_metadata(str(tmp_path), "ai-hub-handler", env=AI_HUB_ENV, models_list_path="")

    data = read_metadata(str(tmp_path))
    assert list(data) == ["models"]
    assert [r["handler"] for r in metadata_records(data)] == ["ai-hub-handler"]


def test_metadata_records_accepts_nothing_but_the_records_list():
    """A document without the list records nothing — same as a missing file."""
    assert metadata_records({"handler": "hf-handler", "model_id": "llamacpp:x"}) == []
    assert metadata_records(None) == []
    assert metadata_records({"models": "not-a-list"}) == []
    assert metadata_records({"models": [{"model_id": "llamacpp:x"}, "junk"]}) == [{"model_id": "llamacpp:x"}]


def test_record_for_file_matches_by_path_or_basename(tmp_path):
    """Patterns match a nested repo layout by path or by name; so must the lookup."""
    _write_quant(tmp_path, "Q4_0", ["Q4_0/model.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])
    data = read_metadata(str(tmp_path))

    assert record_for_file(data, "Q4_0/model.gguf")["files"] == ["Q4_0/model.gguf"]
    assert record_for_file(data, "model.gguf")["files"] == ["Q4_0/model.gguf"]
    assert record_for_file(data, "Qwen3-0.6B-Q8_0.gguf")["files"] == ["Qwen3-0.6B-Q8_0.gguf"]


def test_record_for_file_unclaimed_file_is_unknown(tmp_path):
    """A quantization no surviving record downloaded has no record — never a sibling's."""
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    data = read_metadata(str(tmp_path))
    assert record_for_file(data, "Qwen3-0.6B-Q8_0.gguf") is None
    assert record_for_file(None, "anything.gguf") is None


def test_record_for_file_ignores_a_record_that_names_no_files():
    """A record that says nothing about its files claims no file in particular."""
    data = {"models": [{"handler": "hf-handler", "model_id": "llamacpp:x"}]}
    assert record_for_file(data, "any.gguf") is None


def test_record_for_model_id_selects_the_entrys_record(tmp_path):
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"], model_id="llamacpp:entry-a")
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"], model_id="llamacpp:entry-b")
    data = read_metadata(str(tmp_path))

    assert record_for_model_id(data, "llamacpp:entry-a")["files"] == ["Qwen3-0.6B-Q4_0.gguf"]
    assert record_for_model_id(data, "llamacpp:entry-b")["files"] == ["Qwen3-0.6B-Q8_0.gguf"]
    # A multi-record file answers strictly: no record with the id, no record.
    assert record_for_model_id(data, "llamacpp:entry-c") is None
    assert record_for_model_id(None, "llamacpp:entry-a") is None


def test_record_for_model_id_ignores_a_document_without_records():
    """No ``models`` list, no records — even when a stray top-level id matches."""
    flat = {"handler": "hf-handler", "model_id": "llamacpp:whatever"}
    assert record_for_model_id(flat, "llamacpp:whatever") is None


def test_prune_drops_the_record_of_a_deleted_quantization(tmp_path):
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])
    (tmp_path / "Qwen3-0.6B-Q4_0.gguf").write_bytes(b"\0")  # Q8_0 was deleted

    prune_metadata_records(str(tmp_path))

    data = read_metadata(str(tmp_path))
    assert list(data) == ["models"]
    assert [r["files"] for r in metadata_records(data)] == [["Qwen3-0.6B-Q4_0.gguf"]]


def test_prune_goes_by_the_main_file_not_the_shared_mmproj(tmp_path):
    """The mmproj companion is shared: its presence must not keep a deleted
    quantization's record alive, nor its deletion kill a surviving one's."""
    _write_quant(tmp_path, "Q4_0", ["gemma-Q4_0.gguf", "mmproj-F16.gguf"])
    _write_quant(tmp_path, "Q8_0", ["gemma-Q8_0.gguf", "mmproj-F16.gguf"])
    (tmp_path / "gemma-Q4_0.gguf").write_bytes(b"\0")  # Q8_0 and the mmproj deleted

    prune_metadata_records(str(tmp_path))

    records = metadata_records(read_metadata(str(tmp_path)))
    assert [r["files"] for r in records] == [["gemma-Q4_0.gguf", "mmproj-F16.gguf"]]


def test_prune_removes_the_file_when_no_record_survives(tmp_path):
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    prune_metadata_records(str(tmp_path))
    assert not (tmp_path / METADATA_NAME).exists()


def test_prune_ignores_a_document_without_records(tmp_path):
    """A file holding no ``models`` list records nothing, so nothing is pruned."""
    (tmp_path / METADATA_NAME).write_text("handler: hf-handler\nmodel_id: llamacpp:x\n")
    before = (tmp_path / METADATA_NAME).read_text()
    prune_metadata_records(str(tmp_path))
    assert (tmp_path / METADATA_NAME).read_text() == before


def test_prune_is_a_noop_while_every_file_is_present(tmp_path):
    _write_quant(tmp_path, "Q4_0", ["Qwen3-0.6B-Q4_0.gguf"])
    _write_quant(tmp_path, "Q8_0", ["Qwen3-0.6B-Q8_0.gguf"])
    (tmp_path / "Qwen3-0.6B-Q4_0.gguf").write_bytes(b"\0")
    (tmp_path / "Qwen3-0.6B-Q8_0.gguf").write_bytes(b"\0")
    before = (tmp_path / METADATA_NAME).read_text()

    prune_metadata_records(str(tmp_path))
    assert (tmp_path / METADATA_NAME).read_text() == before


# --------------------------------------------------------------------------- #
# End-to-end payload shape, per handler
# --------------------------------------------------------------------------- #
PAYLOAD_KEYS = ["downloaded_at", "handler", "model_id", "model_origin", "inputs"]


def test_payload_ai_hub(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_metadata(str(model_dir), "ai-hub-handler", env=AI_HUB_ENV, models_list_path=_models_list(tmp_path))
    data = read_metadata(str(model_dir))
    assert list(data) == ["models"]
    record = metadata_records(data)[0]
    assert list(record) == PAYLOAD_KEYS
    assert record["handler"] == "ai-hub-handler"
    assert record["model_id"] == "genie:qwen3_4b_instruct_2507"
    assert record["model_origin"] == "built_in"
    assert record["inputs"] == AI_HUB_ENV


def test_payload_edge_impulse(tmp_path):
    model_dir = tmp_path / "efficientnet-b4-qnn"
    model_dir.mkdir()
    write_metadata(str(model_dir), "ei-handler", env=EI_ENV, models_list_path=_models_list(tmp_path))
    record = metadata_records(read_metadata(str(model_dir)))[0]
    assert list(record) == PAYLOAD_KEYS
    assert record["model_id"] == "ei:efficientnet-b4"
    assert record["inputs"] == EI_ENV
    # YAML ints arrive as strings and are recorded verbatim.
    assert record["inputs"]["ei_project_id"] == "948887"
    # quantization is not set for this model: the key is omitted, not written empty.
    assert "quantization" not in record["inputs"]


def test_payload_hugging_face(tmp_path):
    model_dir = tmp_path / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    model_dir.mkdir(parents=True)
    write_metadata(str(model_dir), "hf-handler", env=HF_ENV, models_list_path=_models_list(tmp_path))
    record = metadata_records(read_metadata(str(model_dir)))[0]
    assert list(record) == PAYLOAD_KEYS
    assert record["model_id"] == "llamacpp:gemma-4-E2B_q4_0-it"
    assert record["inputs"] == HF_ENV
    # The pinned commit is already in the recorded model_url.
    assert "1894d1fc" in record["inputs"]["model_url"]


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
    record = metadata_records(read_metadata(str(model_dir)))[0]
    assert list(record) == PAYLOAD_KEYS
    assert record["model_id"] == "llamacpp:mistral-7b-instruct-v0.2.Q4_0"
    assert record["model_origin"] == "user"
    # The download is still fully described by its own variables.
    assert record["inputs"] == env


def test_written_file_is_plain_safe_yaml(tmp_path):
    """No python-specific tags: the file must be readable by any YAML parser."""
    write_metadata(str(tmp_path), "hf-handler", env=HF_ENV, models_list_path="")
    text = (tmp_path / METADATA_NAME).read_text()
    assert "!!python" not in text
    assert yaml.safe_load(text)["models"][0]["inputs"] == HF_ENV
