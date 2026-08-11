# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Read/write the per-model ".download" in-progress marker.

The marker lives at ``<model_dir>/.download`` and exists only while a download is
in progress (or was interrupted before completing). It is a JSON document so
external tools can match the in-progress model back to models-list.yaml:

    {
      "status": "downloading",
      "handler": "hf-handler",
      "models_repository": "llamacpp",
      "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf",
      "file_patterns": ["*Q4_0*.gguf"]
    }

``file_patterns`` names the files being downloaded, and matters when the directory can
hold more than one model: a Hugging Face repository directory is shared by every
quantization ever downloaded from that repository, so the bare presence of a marker there
does not mean that all of them are in progress. It is optional — a handler whose model
directory holds exactly one model has nothing to disambiguate, and a marker written
before the field existed carries none — and its absence means the whole directory.
"""

import json
import os

MARKER_NAME = ".download"


def marker_payload(handler="", models_repository="", model_directory="", model_url="", file_patterns=None):
    """Build the marker dict from the fields that match a models-list.yaml entry.

    ``model_url`` is included only when set (some handlers download by URL), and so is
    ``file_patterns`` (only handlers sharing a directory between models need it).
    """
    payload = {
        "status": "downloading",
        "handler": handler or "",
        "models_repository": models_repository or "",
        "model_directory": model_directory or "",
    }
    if model_url:
        payload["model_url"] = model_url
    if file_patterns:
        payload["file_patterns"] = list(file_patterns)
    return payload


def write_marker(model_dir, handler="", models_repository="", model_directory="", model_url="", file_patterns=None):
    """Create ``<model_dir>/.download`` with the JSON payload and return its path."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MARKER_NAME)
    with open(path, "w") as f:
        json.dump(marker_payload(handler, models_repository, model_directory, model_url, file_patterns), f)
        f.write("\n")
    return path


def read_marker(path):
    """Parse a marker file into a dict, or None if it is missing / unreadable.

    Backward compatible with the legacy plaintext format (a bare model directory
    name), which is returned as a payload with that value as ``model_directory``.
    """
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if not text:
        return marker_payload()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        return data
    return marker_payload(model_directory=text)
