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
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models_list import load_models_list, MODELS_LIST_PATH


MODELS_BASE_DIR = "/models"


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
            model_size_mb = model_data.get("metadata", {}).get("model_size_mb")

            if not deployment:
                continue

            pre_loaded = deployment.get("pre-loaded", False)

            if pre_loaded:
                results.append({
                    "id": model_id,
                    "name": name,
                    "handler": deployment.get("handler", ""),
                    "model_directory": "",
                    "models_repository": "",
                    "model_type": "",
                    "model_name": "",
                    "model_size_mb": model_size_mb,
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
                    model_directory = variables.get("model_directory") or build_model_directory(variables) or variables.get("model_name", "")
                    models_repository = variables.get("models_repository", "")

                    results.append({
                        "id": model_id,
                        "name": name,
                        "handler": deployment.get("handler", ""),
                        "model_directory": model_directory,
                        "models_repository": models_repository,
                        "model_type": variables.get("model_type", ""),
                        "model_name": variables.get("model_name", ""),
                        "model_size_mb": model_size_mb,
                        "pre_loaded": False,
                        "supported_boards": supported_boards,
                    })

    return results


def get_model_subdir(models_repository):
    """Extract the relative subfolder from models_repository path.

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


def get_dir_size_mb(path):
    """Return total disk usage of a path (file or directory) in MB, rounded to 2 decimals."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return None

    if stat.S_ISREG(st.st_mode):
        return round(st.st_size / 1024 / 1024, 2)
    if not stat.S_ISDIR(st.st_mode):
        return None

    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        # Don't follow symlinks; use cached stat from DirEntry.
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    mode = entry_stat.st_mode
                    if stat.S_ISDIR(mode):
                        stack.append(entry.path)
                    elif stat.S_ISREG(mode):
                        total += entry_stat.st_size
        except OSError:
            continue
    return round(total / 1024 / 1024, 2)


# Cache of os.scandir results keyed by search_dir.
# Each entry is a list of (name, is_dir) tuples, or None if the dir doesn't exist.
_SEARCH_DIR_CACHE = {}


def _scandir_cached(search_dir):
    """Return cached [(name, is_dir), ...] for search_dir, or None if missing."""
    cached = _SEARCH_DIR_CACHE.get(search_dir)
    if cached is not None or search_dir in _SEARCH_DIR_CACHE:
        return cached
    try:
        with os.scandir(search_dir) as it:
            entries = [(e.name, e.is_dir(follow_symlinks=False)) for e in it]
    except (FileNotFoundError, NotADirectoryError):
        entries = None
    except OSError:
        entries = None
    _SEARCH_DIR_CACHE[search_dir] = entries
    return entries


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

    full_path = os.path.join(search_dir, model_directory)
    entries = _scandir_cached(search_dir)
    if entries is None:
        return False, full_path

    # Exact match first (directory or file)
    for name, _is_dir in entries:
        if name == model_directory:
            return True, full_path

    # Check for directories that start with model_directory (e.g. _proxy suffix)
    # Also normalize hyphens/underscores for fuzzy matching
    normalized = model_directory.replace("-", "_")
    for name, is_dir in entries:
        if not is_dir:
            continue
        if name.startswith(model_directory) or name.replace("-", "_").startswith(normalized):
            return True, os.path.join(search_dir, name)

    return False, full_path


def model_is_downloading(model_info, models_base_dir):
    """Return the in-progress model name if a ".download" marker exists, else None.

    The marker is per-model: AI Hub/HF store it inside the model directory
    (<dir>/.download), Edge Impulse stores a sibling (.<file>.download). Its
    presence means the download was interrupted before completing.
    """
    subdir = get_model_subdir(model_info.get("models_repository", ""))
    search_dir = os.path.join(models_base_dir, subdir) if subdir else models_base_dir
    model_directory = model_info.get("model_directory") or ""
    model_name = model_info.get("model_name") or ""
    candidates = []
    if model_directory:
        candidates.append(os.path.join(search_dir, model_directory, ".download"))
    if model_name:
        candidates.append(os.path.join(search_dir, f".{model_name}.download"))
    for marker in candidates:
        try:
            with open(marker) as f:
                return f.read().strip() or True
        except OSError:
            continue
    return None


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
            entry = {
                "id": model_info["id"],
                "name": model_info["name"],
                "handler": model_info["handler"],
                "installed": True,
            }
            if model_info.get("model_size_mb") is not None:
                entry["model_size_mb"] = model_info["model_size_mb"]
        else:
            exists, path = check_model_exists(model_info, args.models_dir)
            # Per-model ".download" marker present => download in progress/incomplete.
            downloading = bool(model_is_downloading(model_info, args.models_dir))
            entry = {
                "id": model_info["id"],
                "name": model_info["name"],
                "handler": model_info["handler"],
                "installed": exists and not downloading,
                "downloading": downloading,
            }
            if model_info.get("model_size_mb") is not None:
                entry["model_size_mb"] = model_info["model_size_mb"]
            if exists:
                entry["path"] = path
                entry["disk_size_mb"] = get_dir_size_mb(path)

        if args.installed_only and not entry["installed"]:
            continue
        if args.not_installed_only and entry["installed"]:
            continue

        results.append(entry)

    # Scan for llamacpp .gguf models on the filesystem
    llamacpp_models = find_llamacpp_models(args.models_dir)
    for m in llamacpp_models:
        if args.not_installed_only:
            continue
        m["disk_size_mb"] = get_dir_size_mb(m["path"])
        results.append(m)

    if args.output_json:
        print(json.dumps({"event": "info", "models": results}, indent=2))
    else:
        installed_count = sum(1 for r in results if r["installed"])
        total_count = len(results)
        print(f"Models: {installed_count}/{total_count} installed\n")
        print(f"{'STATUS':<12} {'SIZE (MB)':<12} {'ID':<45} {'NAME':<40} {'PATH'}")
        print("-" * 152)
        for r in results:
            status = "DOWNLOADING" if r.get("downloading") else ("INSTALLED" if r["installed"] else "NOT FOUND")
            size = (
                f"{r['disk_size_mb']:.2f}"
                if r.get("disk_size_mb") is not None
                else (f"{r['model_size_mb']}" if r.get("model_size_mb") is not None else "-")
            )
            path = r.get("path", "")
            print(f"{status:<12} {size:<12} {r['id']:<45} {r['name']:<40} {path}")


if __name__ == "__main__":
    main()
