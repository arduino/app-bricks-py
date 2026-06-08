# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Shared utilities for loading and querying models-list.yaml."""

import yaml


MODELS_LIST_PATH = "/app/models-list.yaml"


def load_models_list(yaml_path):
    """Load models-list.yaml and return the list of model entries."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def find_model_size_mb(models, model_key):
    """Return model_size_mb for the given model_key, or -1 if not found."""
    for entry in models:
        if not isinstance(entry, dict):
            continue
        for entry_key, model_data in entry.items():
            if entry_key == model_key and isinstance(model_data, dict):
                metadata = model_data.get("metadata", {})
                return metadata.get("model_size_mb", -1)
    return -1
