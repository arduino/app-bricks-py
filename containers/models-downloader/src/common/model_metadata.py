# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Write/read the per-model ".arduino_metadata.yaml" record of a completed download.

The downloaders receive their whole input as lowercase environment variables, taken
from ``deployment.platforms.<board>.variables`` in models-list.yaml, and the
in-progress ``.download`` marker is deleted once the download succeeds — so without
this file nothing on disk would record *what* was downloaded. It is written inside
the model directory after a successful download and stays there:

    schema_version: 1
    downloaded_at: '2026-08-03T09:41:12Z'
    handler: hf-handler
    model_id: llamacpp:gemma-4-E2B_q4_0-it
    model_id_source: models-list
    inputs:                     # the download variables, verbatim from the environment
      models_repository: llamacpp
      model_directory: google/gemma-4-E2B-it-qat-q4_0-gguf
      model_url: https://huggingface.co/google/...

Contracts callers must honour:

- ``write_metadata`` never raises and never fails a download. Bookkeeping must not
  turn a completed multi-GB transfer into a failure, so a write error is reported as
  an ``info`` event (not ``error``, which the host would read as a failed download)
  and the function returns None.
- The file is therefore **optional**. Its absence means "unknown / legacy install",
  never "up to date": models downloaded before this file existed have none.
- ``model_id: null`` with ``model_id_source: unresolved`` is a **normal, supported
  state, not an error**. Any Hugging Face repository can be downloaded ad hoc via
  ``--model-key`` / ``--model-repo-id`` / ``--model-url`` without a models-list.yaml
  entry; such a download is recorded in full, only unidentified. Consumers must not
  treat a null model_id as a failure, and outdated-detection simply does not apply
  (there is no declaration to compare against).
- Nothing else is copied out of models-list.yaml. ``model_id`` points back at the
  entry, and every other field of it (name, description, source, model_size_mb, ...)
  is read from models-list.yaml itself rather than duplicated — and left to go stale —
  here. The record holds only what models-list.yaml cannot tell you: which variables
  this install was actually downloaded with, and when.
- The Hugging Face handler downloads into a per-*repository* directory, so two
  models-list.yaml entries pulling different quantizations out of the same repo would
  share one metadata file (the last download wins). The ``.download`` marker and the
  "Model exists" early return have the same limitation today.
"""

import json
import os
from datetime import datetime, timezone

import yaml

from common.models_list import MODELS_LIST_PATH, find_matching_model, load_models_list

METADATA_NAME = ".arduino_metadata.yaml"
SCHEMA_VERSION = 1

_HEADER = (
    "# Written by the Arduino models-downloader after a successful download.\n"
    "# Do not edit: it records what was downloaded and lets tooling detect outdated models.\n"
)

# Every variable name used in models-list.yaml deployment.platforms.*.variables.
# The order here is the order of the keys in the written "inputs" block.
INPUT_VARIABLES = (
    "models_repository",
    "model_directory",
    "model_name",
    "model_type",
    "quantization",
    # ai-hub
    "chipset",
    "version",
    # edge impulse
    "ei_project_id",
    "ei_impulse_id",
    "target",
    # hugging face — model_url carries either a file URL or a compact model key
    "model_url",
    "model_mmproj_url",
)

# Credentials are never persisted, whatever the environment holds.
SECRET_VARIABLES = frozenset({"hf_token", "HF_TOKEN", "HF_HUB_TOKEN", "EI_API_KEY"})


def utc_now_iso():
    """Return the current UTC time as a second-resolution ISO-8601 string ("...Z")."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_bookkeeping_name(name):
    """True when *name* is a metadata/marker file rather than model content.

    Matches the ``.arduino_metadata.yaml.tmp`` sibling of an interrupted atomic
    write too, so a directory holding only that is still treated as incomplete.
    """
    return name == ".download" or name.startswith(METADATA_NAME)


def collect_inputs(env=None, extra_keys=()):
    """Return the download variables present in *env*, in INPUT_VARIABLES order.

    Empty values are dropped (an unset variable arrives as an empty string, and
    recording ``quantization: ''`` would be indistinguishable from a real value),
    as are credentials. Values are kept verbatim as strings.
    """
    env = env if env is not None else os.environ
    inputs = {}
    for key in tuple(INPUT_VARIABLES) + tuple(extra_keys):
        if key in inputs or key in SECRET_VARIABLES:
            continue
        value = env.get(key)
        if value:
            inputs[key] = value
    return inputs


def identify_model(env=None, models_list_path=MODELS_LIST_PATH):
    """Identify which models-list.yaml entry *env* describes.

    The host does not pass the entry's map key as an environment variable, so it is
    recovered by matching the download variables against the models-list.yaml baked
    into the image. A ``model_id`` variable is honoured first, so the day the host
    starts providing one this lookup is bypassed.

    Returns:
        A dict with ``model_id`` (None when unidentified) and ``model_id_source``
        ("env", "models-list" or "unresolved"). Nothing else is taken from the entry:
        the id is the pointer back to it, see the module docstring.
    """
    env = env if env is not None else os.environ

    explicit = env.get("model_id")
    if explicit:
        return {"model_id": explicit, "model_id_source": "env"}

    try:
        models = load_models_list(models_list_path)
        model_id, _model_data, _platform = find_matching_model(models, env, board=env.get("BOARD_NAME"))
    except Exception:  # noqa: BLE001 - a missing or broken models-list.yaml must not matter
        return {"model_id": None, "model_id_source": "unresolved"}

    if not model_id:
        return {"model_id": None, "model_id_source": "unresolved"}
    return {"model_id": model_id, "model_id_source": "models-list"}


def metadata_payload(handler, inputs=None, identity=None, downloaded_at=None):
    """Build the metadata document, dropping empty ``inputs`` entries."""
    identity = identity or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "downloaded_at": downloaded_at or utc_now_iso(),
        "handler": handler or "",
        "model_id": identity.get("model_id"),
        "model_id_source": identity.get("model_id_source", "unresolved"),
    }
    kept = {k: v for k, v in (inputs or {}).items() if v is not None and v != ""}
    if kept:
        payload["inputs"] = kept
    return payload


def write_metadata(model_dir, handler, env=None, models_list_path=MODELS_LIST_PATH, extra_input_keys=()):
    """Write ``<model_dir>/.arduino_metadata.yaml`` atomically; return its path or None.

    Called after a successful download and *before* clearing the ``.download``
    marker: if this process dies in between, the marker still marks the directory as
    incomplete and the next run wipes and retries, so no directory can end up
    installed-but-unrecorded. Never raises — see the module docstring.
    """
    path = os.path.join(model_dir, METADATA_NAME)
    tmp = path + ".tmp"
    try:
        payload = metadata_payload(
            handler,
            inputs=collect_inputs(env, extra_input_keys),
            identity=identify_model(env, models_list_path),
        )
        os.makedirs(model_dir, exist_ok=True)
        with open(tmp, "w") as f:
            f.write(_HEADER)
            # width: keep long URLs and commands on one line rather than folded.
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=4096)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path
    except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a completed download
        try:
            os.unlink(tmp)
        except OSError:
            pass
        # Deliberately "info": an "error" event would make the host report the
        # finished download as failed.
        print(json.dumps({"event": "info", "description": f"Could not write {METADATA_NAME}: {exc}"}), flush=True)
        return None


def read_metadata(path):
    """Parse a metadata file into a dict, or None if it is missing / unusable.

    *path* may be the file itself or the model directory containing it. Anything
    unreadable, malformed or not a mapping yields None, and ``schema_version`` is
    deliberately not checked so a newer file never breaks an older reader.
    """
    if os.path.isdir(path):
        path = os.path.join(path, METADATA_NAME)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None
