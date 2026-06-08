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


def find_model_size_mb(models, model_type, model_name):
    """Return model_size_mb for the model whose deployment variables match model_type and model_name, or -1 if not found."""
    for entry in models:
        if not isinstance(entry, dict):
            continue
        for _entry_key, model_data in entry.items():
            if not isinstance(model_data, dict):
                continue
            deployment = model_data.get("deployment", {})
            platforms = deployment.get("platforms", [])
            for platform_entry in platforms:
                if not isinstance(platform_entry, dict):
                    continue
                for _platform_name, platform_config in platform_entry.items():
                    variables = platform_config.get("variables", {}) if isinstance(platform_config, dict) else {}
                    if variables.get("model_type") == model_type and variables.get("model_name") == model_name:
                        metadata = model_data.get("metadata", {})
                        return metadata.get("model_size_mb", -1)
    return -1
