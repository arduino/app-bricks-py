# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Write/read the per-model ".arduino_metadata.yaml" record of a completed download.

The downloaders receive their whole input as lowercase environment variables, taken
from ``deployment.platforms.<board>.variables`` in models-list.yaml, and the
in-progress ``.download`` marker is deleted once the download succeeds — so without
this file nothing on disk would record *what* was downloaded. It is written inside
the model directory after a successful download and stays there:

    downloaded_at: '2026-08-03T09:41:12Z'
    handler: hf-handler
    model_id: llamacpp:gemma-4-E2B_q4_0-it
    model_origin: builtin
    inputs:                     # the download variables, verbatim from the environment
      models_repository: llamacpp
      model_directory: google/gemma-4-E2B-it-qat-q4_0-gguf
      model_url: https://huggingface.co/google/...

Contracts callers must honour:

- ``write_metadata`` never raises: a write error is reported as an ``info`` event and
  the function returns None. Whether that fails the download is the caller's decision,
  and the two kinds of handler decide it differently:

  * The Hugging Face handler treats None as **fatal** — it keeps the ".download"
    marker (so the next run discards and retries) and exits non-zero. This reverses
    the original "bookkeeping must never fail a completed multi-GB transfer" stance,
    deliberately: an ad-hoc model is deleted through the API by the ``inputs``
    recorded here, so an installed-but-unrecorded ad-hoc model would be permanently
    unmanageable, while a lost transfer can simply be repeated. The reversal happened
    before ad-hoc models went GA, which is what makes the next bullet possible.
  * The AI Hub and Edge Impulse handlers keep it best-effort: their models are all
    declared in models-list.yaml, so the host can always manage them from the
    declaration alone and the record only feeds outdated-detection.

- For **readers** the file stays optional — installs made before it existed, or by a
  best-effort handler, have none, and its absence means "unknown / legacy install",
  never "up to date". But every ad-hoc model installed by a current downloader is
  guaranteed to carry one.
- ``model_origin`` says where the model comes from, not where its id was read from:
  ``builtin`` for a model models-list.yaml declares, ``user_configured`` for one
  downloaded ad hoc. Any Hugging Face repository can be fetched by putting its URL or
  compact key in ``model_url`` with no entry in the list, so ``user_configured`` is a
  **normal, supported state, not an error**. Only that outdated-detection does not
  apply to it: there is no declaration to compare against. The listing reports the
  same field with the same two values.
- ``model_id`` is always set for a model that downloaded successfully, curated or not.
  A user-configured model has no entry key to borrow, so the handler supplies a
  ``fallback_model_id`` derived from what it fetched, matching the id the listing
  reports for the same files (``common/gguf_naming.py``). That id is a snapshot: if
  the model is later added to the curated catalog its listing name changes to the
  entry's stem-form id and the recorded one goes stale. The listing, derived from
  the filesystem and the catalog, is the authority.
- Nothing else is copied out of models-list.yaml. ``model_id`` points back at the
  entry, and every other field of it (name, description, source, model_size_mb, ...)
  is read from models-list.yaml itself rather than duplicated — and left to go stale —
  here. The record holds only what models-list.yaml cannot tell you: which variables
  this install was actually downloaded with, and when.
- The Hugging Face handler downloads into a per-*repository* directory, so two
  models-list.yaml entries pulling different quantizations out of the same repo share
  one metadata file: the last download wins, and the quantization downloaded first is
  left described by a record naming the other one. Only this file is shared — whether a
  model is installed, and what an interrupted download discards, are decided per
  quantization, and the ``.download`` marker says which files it stands for.
"""

import json
import os
from datetime import datetime, timezone

import yaml

from common.models_list import MODELS_LIST_PATH, find_matching_model, load_models_list

METADATA_NAME = ".arduino_metadata.yaml"

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

# Where a model comes from, reported as "model_origin" both here and in the listing
# (list_models.py imports these, so the two can never drift apart).
ORIGIN_BUILTIN = "builtin"
ORIGIN_USER_CONFIGURED = "user_configured"


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


def identify_model(env=None, models_list_path=MODELS_LIST_PATH, fallback_model_id=None):
    """Identify the model *env* describes, and say where that model comes from.

    The host does not pass the entry's map key as an environment variable, so a curated
    model is recognised by matching the download variables against the models-list.yaml
    baked into the image. A ``model_id`` variable is honoured first, so the day the host
    starts providing one this lookup is bypassed.

    No match means the model is not curated — a user-configured download — which is a
    normal outcome, not a failure. Such a model still needs an id to be referred to by,
    so *fallback_model_id* (derived by the handler from what it actually fetched) is
    used instead of leaving it null.

    Returns:
        A dict with ``model_id`` and ``model_origin``, the latter being
        ``ORIGIN_BUILTIN`` or ``ORIGIN_USER_CONFIGURED``.
    """
    env = env if env is not None else os.environ

    explicit = env.get("model_id")
    if explicit:
        # Only models-list.yaml can tell the host an id, so this is a curated model.
        return {"model_id": explicit, "model_origin": ORIGIN_BUILTIN}

    try:
        models = load_models_list(models_list_path)
        model_id, _model_data, _platform = find_matching_model(models, env, board=env.get("BOARD_NAME"))
    except Exception:  # noqa: BLE001 - a missing or broken models-list.yaml must not matter
        model_id = None

    if model_id:
        return {"model_id": model_id, "model_origin": ORIGIN_BUILTIN}
    return {"model_id": fallback_model_id, "model_origin": ORIGIN_USER_CONFIGURED}


def metadata_payload(handler, inputs=None, identity=None, downloaded_at=None):
    """Build the metadata document, dropping empty ``inputs`` entries."""
    identity = identity or {}
    payload = {
        "downloaded_at": downloaded_at or utc_now_iso(),
        "handler": handler or "",
        "model_id": identity.get("model_id"),
        "model_origin": identity.get("model_origin", ORIGIN_USER_CONFIGURED),
    }
    kept = {k: v for k, v in (inputs or {}).items() if v is not None and v != ""}
    if kept:
        payload["inputs"] = kept
    return payload


def write_metadata(model_dir, handler, env=None, models_list_path=MODELS_LIST_PATH, extra_input_keys=(), fallback_model_id=None):
    """Write ``<model_dir>/.arduino_metadata.yaml`` atomically; return its path or None.

    Called after a successful download and *before* clearing the ``.download``
    marker: if this process dies in between, the marker still marks the directory as
    incomplete and the next run wipes and retries, so no directory can end up
    installed-but-unrecorded. Never raises — see the module docstring.

    *fallback_model_id* names the model when models-list.yaml does not declare it; pass
    one whenever the handler can download something the list has never heard of.
    """
    path = os.path.join(model_dir, METADATA_NAME)
    tmp = path + ".tmp"
    try:
        payload = metadata_payload(
            handler,
            inputs=collect_inputs(env, extra_input_keys),
            identity=identify_model(env, models_list_path, fallback_model_id),
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
    unreadable, malformed or not a mapping yields None. Whatever keys the mapping holds
    are returned as they are, so a file written by a newer version — with fields this
    reader knows nothing about — never breaks it.
    """
    if os.path.isdir(path):
        path = os.path.join(path, METADATA_NAME)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None
