# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""List all models and their presence on the filesystem.

Reads models-list.yaml and checks whether each model with a deployment
section is present under /models (or a custom base path).

Usage:
    python list_models.py
    python list_models.py --models-dir /custom/models
    python list_models.py --model-list /path/to/models-list.yaml
    python list_models.py --json
"""

import argparse
import json
import os
import sys

import yaml


MODELS_LIST_PATH = "/app/models-list.yaml"
MODELS_BASE_DIR = "/models"


def load_models_list(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def get_model_info(model_entry):
    """Extract model id, name, and filesystem paths from a model entry."""
    results = []

    for item in model_entry if isinstance(model_entry, list) else [model_entry]:
        if not isinstance(item, dict):
            continue
        for model_id, model_data in item.items():
            if not isinstance(model_data, dict):
                continue

            name = model_data.get("name", model_id)
            supported_boards = model_data.get("supported_boards", [])
            deployment = model_data.get("deployment")

            if not deployment:
                continue

            pre_loaded = deployment.get("pre-loaded", False)

            if pre_loaded:
                results.append({
                    "id": model_id,
                    "name": name,
                    "handler": deployment.get("handler", ""),
                    "platform": "",
                    "model_directory": "",
                    "models_repository": "",
                    "model_type": "",
                    "model_name": "",
                    "pre_loaded": True,
                    "supported_boards": supported_boards,
                })
                continue

            if "platforms" not in deployment:
                continue

            for platform_entry in deployment["platforms"]:
                if not isinstance(platform_entry, dict):
                    continue
                for platform_name, platform_config in platform_entry.items():
                    variables = platform_config.get("variables", {})
                    model_directory = variables.get("model_directory") or build_model_directory(variables)
                    models_repository = variables.get("models_repository", "")

                    results.append({
                        "id": model_id,
                        "name": name,
                        "handler": deployment.get("handler", ""),
                        "platform": platform_name,
                        "model_directory": model_directory,
                        "models_repository": models_repository,
                        "model_type": variables.get("model_type", ""),
                        "model_name": variables.get("model_name", ""),
                        "pre_loaded": False,
                        "supported_boards": supported_boards,
                    })

    return results


def get_model_subdir(models_repository):
    """Extract the relative subfolder from models_repository path.

    e.g. "/var/lib/arduino-app-cli/models/audio-analytics/tts" -> "audio-analytics/tts"
         "/var/lib/arduino-app-cli/models/genai" -> "genai"
    """
    marker = "/models/"
    idx = models_repository.rfind(marker)
    if idx != -1:
        return models_repository[idx + len(marker) :]
    return ""


def build_model_directory(variables):
    """Build model_directory from variables when not explicitly set.

    Pattern: {model_name}-{model_type}-{quantization}-{chipset}
    """
    model_name = variables.get("model_name", "")
    model_type = variables.get("model_type", "")
    quantization = variables.get("quantization", "")
    chipset = variables.get("chipset", "")
    if model_name and model_type and quantization and chipset:
        return f"{model_name}-{model_type}-{quantization}-{chipset}"
    return ""


def check_model_exists(model_info, models_base_dir):
    """Check if a model exists on the filesystem."""
    model_directory = model_info.get("model_directory") or ""
    if not model_directory:
        return False, ""

    # Build full path using models_repository subfolder
    subdir = get_model_subdir(model_info.get("models_repository", ""))
    if subdir:
        search_dir = os.path.join(models_base_dir, subdir)
    else:
        search_dir = models_base_dir

    # Exact match first
    full_path = os.path.join(search_dir, model_directory)
    if os.path.isdir(full_path):
        return True, full_path

    # Check for directories that start with model_directory (e.g. _proxy suffix)
    # Also normalize hyphens/underscores for fuzzy matching
    if os.path.isdir(search_dir):
        normalized = model_directory.replace("-", "_")
        for entry in os.listdir(search_dir):
            entry_normalized = entry.replace("-", "_")
            if (entry.startswith(model_directory) or entry_normalized.startswith(normalized)) and os.path.isdir(os.path.join(search_dir, entry)):
                return True, os.path.join(search_dir, entry)

    return False, full_path


LLAMACPP_SUBDIR = "llamacpp"


def find_llamacpp_models(models_base_dir):
    """Scan for .gguf files under the llamacpp directory."""
    llamacpp_dir = os.path.join(models_base_dir, LLAMACPP_SUBDIR)
    results = []
    if not os.path.isdir(llamacpp_dir):
        return results

    for root, _dirs, files in os.walk(llamacpp_dir):
        for f in files:
            if f.endswith(".gguf"):
                full_path = os.path.join(root, f)
                model_name = os.path.splitext(f)[0]
                results.append({
                    "id": f"llamacpp:{model_name}",
                    "name": model_name,
                    "handler": "llamacpp",
                    "platform": "",
                    "model_type": "llamacpp",
                    "path": full_path,
                    "installed": True,
                })
    return results


def main():
    parser = argparse.ArgumentParser(description="List all models and their filesystem status.")
    parser.add_argument(
        "--models-dir",
        default=MODELS_BASE_DIR,
        help=f"Base directory where models are mounted (default: {MODELS_BASE_DIR}).",
    )
    parser.add_argument(
        "--model-list",
        default=MODELS_LIST_PATH,
        dest="yaml_path",
        help=f"Path to models-list.yaml (default: {MODELS_LIST_PATH}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="Only show models that are installed.",
    )
    parser.add_argument(
        "--not-installed-only",
        action="store_true",
        help="Only show models that are NOT installed.",
    )
    parser.add_argument(
        "--supported-board",
        type=str,
        metavar="BOARD",
        help="Filter models by supported board (e.g. ventunoq). Models without a supported_boards entry are always included.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.yaml_path):
        print(json.dumps({"event": "error", "description": f"models-list.yaml not found at {args.yaml_path}"}))
        sys.exit(1)

    models_list = load_models_list(args.yaml_path)
    all_models = []
    for entry in models_list:
        all_models.extend(get_model_info(entry))

    # Filter by supported board
    if args.supported_board:
        all_models = [m for m in all_models if not m["supported_boards"] or args.supported_board in m["supported_boards"]]

    results = []
    for model_info in all_models:
        if model_info.get("pre_loaded"):
            exists = True
            path = "pre-loaded"
        else:
            exists, path = check_model_exists(model_info, args.models_dir)

        if args.installed_only and not exists:
            continue
        if args.not_installed_only and exists:
            continue

        results.append({
            "id": model_info["id"],
            "name": model_info["name"],
            "handler": model_info["handler"],
            "platform": model_info["platform"],
            "model_type": model_info["model_type"],
            "path": path,
            "installed": exists,
        })

    # Scan for llamacpp .gguf models on the filesystem
    llamacpp_models = find_llamacpp_models(args.models_dir)
    for m in llamacpp_models:
        if args.not_installed_only:
            continue
        results.append(m)

    if args.output_json:
        print(json.dumps({"event": "info", "models": results}, indent=2))
    else:
        installed_count = sum(1 for r in results if r["installed"])
        total_count = len(results)
        print(f"Models: {installed_count}/{total_count} installed\n")
        print(f"{'STATUS':<12} {'ID':<45} {'NAME':<40} {'PATH'}")
        print("-" * 140)
        for r in results:
            status = "INSTALLED" if r["installed"] else "NOT FOUND"
            print(f"{status:<12} {r['id']:<45} {r['name']:<40} {r['path']}")


if __name__ == "__main__":
    main()
