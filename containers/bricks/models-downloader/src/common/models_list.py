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


def get_model_subdir(models_repository):
    """Extract the relative subfolder from a models_repository path.

    e.g. "/var/lib/arduino-app-cli/models/audio-analytics/tts" -> "audio-analytics/tts"
         "/var/lib/arduino-app-cli/models/genai" -> "genai"
         "models/genai" -> "genai"
         "models/audio-analytics/asr" -> "audio-analytics/asr"
    """
    marker = "/models/"
    idx = models_repository.rfind(marker)
    if idx != -1:
        return models_repository[idx + len(marker) :]
    # Handle relative paths like "models/genai" or "models/audio-analytics/asr"
    if models_repository.startswith("models/"):
        return models_repository[len("models/") :]
    # Bare repository name (e.g. "edge-impulse", "genai") => use as-is
    if models_repository:
        return models_repository
    return ""


def _iter_platform_variables(models):
    """Yield ``(model_id, model_data, platform_name, variables)`` for every deployment platform."""
    for entry in models:
        if not isinstance(entry, dict):
            continue
        for model_id, model_data in entry.items():
            if not isinstance(model_data, dict):
                continue
            deployment = model_data.get("deployment") or {}
            if not isinstance(deployment, dict):
                continue
            for platform_entry in deployment.get("platforms") or []:
                if not isinstance(platform_entry, dict):
                    continue
                for platform_name, platform_config in platform_entry.items():
                    if not isinstance(platform_config, dict):
                        continue
                    variables = platform_config.get("variables") or {}
                    if isinstance(variables, dict):
                        yield model_id, model_data, platform_name, variables


def find_matching_model(models, env, board=None):
    """Find the models-list.yaml entry whose deployment variables match *env*.

    A platform matches when every variable it declares is present in *env* with an
    equal value, compared as strings: YAML ints such as ``ei_project_id: 948887``
    arrive in the environment as ``"948887"``. Extra environment keys are ignored,
    which is what makes the match work at all (the container also sees PATH, HOME,
    ...). Keep ``version`` values quoted in models-list.yaml: an unquoted ``0.51``
    would be parsed as a float and stringify to ``"0.51"``.

    Entries deployed as ``pre-loaded`` have no ``platforms`` and are skipped, as
    are platforms declaring no variables (an empty map matches every environment).

    Args:
        models: The list of entries returned by ``load_models_list``.
        env: Mapping of environment variables (values are strings).
        board: Optional board name, used to pick between platforms of the same entry.

    Returns:
        ``(model_id, model_data, platform_name)``, or ``(None, None, None)`` when
        nothing matches or several different models match equally well.
    """
    matches = []
    for model_id, model_data, platform_name, variables in _iter_platform_variables(models):
        if not variables:
            continue
        if all(str(value) == env.get(key) for key, value in variables.items()):
            # Number of matched variables = how specific this match is.
            matches.append((model_id, model_data, platform_name, len(variables)))

    if not matches:
        return None, None, None

    if board:
        preferred = [m for m in matches if m[2] == board]
        if preferred:
            matches = preferred

    # Several platforms of the *same* entry may match: repeated per-board blocks
    # with identical variables are common, and they all name the same model.
    if len({m[0] for m in matches}) > 1:
        best = max(m[3] for m in matches)
        matches = [m for m in matches if m[3] == best]
        if len({m[0] for m in matches}) > 1:
            return None, None, None  # genuinely ambiguous: better no id than a wrong one

    model_id, model_data, platform_name, _ = matches[0]
    return model_id, model_data, platform_name


def find_model_size_mb(models, model_type, model_name):
    """Return model_size_mb for the model whose deployment variables match model_type and model_name, or -1 if not found."""
    for _model_id, model_data, _platform_name, variables in _iter_platform_variables(models):
        if variables.get("model_type") == model_type and variables.get("model_name") == model_name:
            metadata = model_data.get("metadata", {})
            return metadata.get("model_size_mb", -1)
    return -1
