# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""List all models and their presence on the filesystem.

Reads models-list.yaml and checks whether each model with a deployment
section is present under /models (or a custom base path). GGUF models found under
/models/llamacpp that no entry declares are listed too, since any Hugging Face
repository can be downloaded ad hoc; ``model_origin`` says which of the two a listed
model is ("builtin" or "user_configured").

Usage:
    python list_models.py
    python list_models.py --models-dir /custom/models
    python list_models.py --model-list /path/to/models-list.yaml
    python list_models.py --json
"""

import argparse
import fnmatch
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.download_marker import read_marker
from common.gguf_naming import LLAMACPP_SUBDIR, declaration_covers, declared_gguf_files, gguf_basename, gguf_model_name
from common.model_metadata import (
    ORIGIN_BUILTIN,
    ORIGIN_USER_CONFIGURED,
    file_record,
    is_bookkeeping_name,
    read_metadata,
    record_for_model_id,
)
from common.models_list import get_model_subdir, load_models_list, MODELS_LIST_PATH


MODELS_BASE_DIR = "/models"

# "model_origin" says where a listed model comes from, using the same two values the
# ".arduino_metadata.yaml" record does. A model declared in models-list.yaml is curated
# and its variables can be compared against to detect an outdated install; one found
# only on disk was downloaded ad hoc and there is nothing to compare it to.


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
                    model_directory = (
                        variables.get("model_directory") or build_model_directory(variables) or os.path.splitext(variables.get("model_name", ""))[0]
                    )
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
                        "platform": platform_name,  # Kept for the per-board dedup; never part of the output.
                        "variables": variables,  # Kept for the outdated check; never part of the output.
                    })

    return results


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


def model_search_dir(model_info, models_base_dir):
    """Return the directory that holds the model folder, per its models_repository."""
    subdir = get_model_subdir(model_info.get("models_repository", ""))
    return os.path.join(models_base_dir, subdir) if subdir else models_base_dir


def _has_model_content(path):
    """True when *path* is a file, or a directory holding more than bookkeeping files.

    A folder left with only a ".download" marker or a ".arduino_metadata.yaml" record
    (model files deleted, or a download that never got started) is not an install.
    """
    if os.path.isfile(path):
        return True
    try:
        with os.scandir(path) as it:
            return any(not is_bookkeeping_name(entry.name) for entry in it)
    except OSError:
        return False


def check_model_exists(model_info, models_base_dir):
    """Check if a model exists on the filesystem."""
    model_directory = model_info.get("model_directory") or ""
    if not model_directory:
        return False, ""

    # Build full path using models_repository subfolder
    search_dir = model_search_dir(model_info, models_base_dir)

    full_path = os.path.join(search_dir, model_directory)
    entries = _scandir_cached(search_dir)
    if entries is None:
        return False, full_path

    # Exact match first (directory or file)
    for name, _is_dir in entries:
        if name == model_directory:
            return _has_model_content(full_path), full_path

    # Check for directories that start with model_directory (e.g. _proxy suffix)
    # Also normalize hyphens/underscores for fuzzy matching
    normalized = model_directory.replace("-", "_")
    for name, is_dir in entries:
        if not is_dir:
            continue
        if name.startswith(model_directory) or name.replace("-", "_").startswith(normalized):
            matched = os.path.join(search_dir, name)
            return _has_model_content(matched), matched

    return False, full_path


def model_is_downloading(model_info, models_base_dir):
    """Return the parsed ".download" marker dict if present, else None.

    The marker is per-model and lives inside the model directory
    (<dir>/.download) for AI Hub, HF and Edge Impulse alike. Its presence means
    a download is in progress or was interrupted before completing.
    """
    search_dir = model_search_dir(model_info, models_base_dir)
    model_directory = model_info.get("model_directory") or ""
    candidates = []
    if model_directory:
        candidates.append(os.path.join(search_dir, model_directory, ".download"))
    for marker in candidates:
        data = read_marker(marker)
        if data is not None:
            return data
    return None


def model_metadata(model_info, models_base_dir, path=None):
    """Return this model's download record from ".arduino_metadata.yaml", or None.

    Written by the downloaders once a download completes, so its absence means the
    model was installed before this record existed (or is not installed at all) —
    never that it is up to date. A Hugging Face repository directory holds one record
    per downloaded quantization, so the entry's own record is picked by its model id
    (``record_for_model_id``): a directory recording only other entries' downloads
    says nothing about this one.

    The canonical folder built from the models-list.yaml variables is tried first,
    because it is the only one that resolves for the nested model_directory of a
    Hugging Face model; *path* covers the folder actually matched on disk, whose
    name may differ (e.g. a "_proxy" suffix).
    """
    model_directory = model_info.get("model_directory") or ""
    if model_directory:
        data = read_metadata(os.path.join(model_search_dir(model_info, models_base_dir), model_directory))
        if data is not None:
            return record_for_model_id(data, model_info.get("id"))
    if path:
        data = read_metadata(path if os.path.isdir(path) else os.path.dirname(path))
        return record_for_model_id(data, model_info.get("id"))
    return None


def outdated_fields(model_info, metadata):
    """Return the download variables that changed in models-list.yaml since download.

    A non-empty result means the installed model no longer matches its declaration
    (a bumped AI Hub ``version``, a new Hugging Face revision in ``model_url``, a new
    Edge Impulse impulse id, ...) and should be downloaded again.
    """
    variables = model_info.get("variables") or {}
    inputs = metadata.get("inputs") or {}
    return sorted(key for key, value in variables.items() if str(value) != inputs.get(key))


def llamacpp_name_from_marker(marker, root):
    """Derive a model name for an in-progress llamacpp download from its marker.

    If the marker carries a ``model_url`` pointing at a .gguf file, the file
    stem is used (e.g. ".../gemma-4-E2B_q4_0-it.gguf" -> "gemma-4-E2B_q4_0-it").
    Otherwise fall back to the model_directory / folder name.
    """
    url = marker.get("model_url") or ""
    if url:
        base = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        if base.endswith(".gguf"):
            return os.path.splitext(base)[0]
        if base:
            return base
    model_directory = marker.get("model_directory") or ""
    if model_directory:
        return os.path.basename(model_directory.rstrip("/"))
    return os.path.basename(root.rstrip(os.sep))


def marker_covers_file(marker, filename):
    """True when *marker* describes an in-progress download of *filename*.

    The marker sits in the model directory, which for Hugging Face is the whole
    repository — and one repository publishes several quantizations, all downloaded into
    that same directory. Only the files ``file_patterns`` names are being downloaded, so
    a quantization installed earlier keeps counting as installed while another one is
    fetched next to it. A marker without the field — another handler, or one written
    before it existed — covers its whole directory, as it always did.
    """
    patterns = marker.get("file_patterns")
    if not isinstance(patterns, list) or not patterns:
        return True
    return any(isinstance(p, str) and fnmatch.fnmatch(filename, p) for p in patterns)


def find_llamacpp_models(models_base_dir, declarations=()):
    """Scan for .gguf models under the llamacpp directory.

    Mirrors the grouping used when generating models.ini: ``*mmproj*.gguf``
    files are not standalone models but the multimodal projection belonging to
    the main GGUF in the same directory, so they are not listed separately.

    Models are named with ``gguf_model_name`` from their ".arduino_metadata.yaml"
    download record, mirroring the models.ini sections: the path-qualified name for
    a user-configured model, the file stem for everything else — each downloaded
    file is one model with its own id, path and size, never merged with a
    same-named stranger. *declarations* (``declared_gguf_files`` of models-list.yaml)
    only name what has no record yet: a pending download, and a file landed by a
    download still in flight.

    A download with no GGUF on disk yet (only a ".download" marker) is still surfaced,
    using the marker info for the entry — including when the directory does hold other
    GGUFs, since those are other quantizations rather than the pending model.

    Every entry carries the private ``_rel_dir``/``_filename`` location keys main()
    matches against the models-list.yaml declarations and strips before output.
    """
    llamacpp_dir = os.path.join(models_base_dir, LLAMACPP_SUBDIR)
    results = []
    if not os.path.isdir(llamacpp_dir):
        return results

    for root, _dirs, files in os.walk(llamacpp_dir):
        gguf_files = sorted(f for f in files if f.endswith(".gguf"))
        marker = read_marker(os.path.join(root, ".download"))
        if not gguf_files and marker is None:
            continue
        rel_dir = os.path.relpath(root, llamacpp_dir).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        mmproj_files = [f for f in gguf_files if "mmproj" in f]

        # Whether the marker was accounted for by a GGUF listed below; if it was not, the
        # download it stands for has nothing on disk yet and is reported on its own.
        marker_listed = False
        for f in gguf_files:
            if "mmproj" in f:
                continue
            downloading = marker is not None and marker_covers_file(marker, f)
            marker_listed = marker_listed or downloading
            full_path = os.path.join(root, f)
            rel_path = f"{rel_dir}/{f}" if rel_dir else f
            # The record can sit above a nested per-quantization folder, so it is
            # looked up from the file's directory up to the llamacpp root. The
            # directory is shared by every quantization of a repository: the file is
            # described by its own download's record, never a sibling's.
            record = file_record(full_path, llamacpp_dir)
            model_name = gguf_model_name(rel_path, record)
            if record is None and downloading:
                # Landed by a download still in flight: its record is written last,
                # so predict the finished id the way the pending branch below does —
                # a declared location keeps the stem, anything else is qualified.
                if not any(declaration_covers(d, n, rel_dir, f) for d, n, _mid in declarations):
                    model_name = rel_path[: -len(".gguf")]
            disk_size_mb = get_dir_size_mb(full_path)
            # The mmproj file in the same directory is part of this model.
            if mmproj_files:
                mmproj_size = get_dir_size_mb(os.path.join(root, mmproj_files[0]))
                if disk_size_mb is not None and mmproj_size is not None:
                    disk_size_mb = round(disk_size_mb + mmproj_size, 2)
            entry = {
                "id": f"llamacpp:{model_name}",
                "name": model_name,
                "handler": "llamacpp",
                # Found on disk. main() overrides this when the id matches a
                # models-list.yaml entry, which makes it a curated model instead.
                "model_origin": ORIGIN_USER_CONFIGURED,
                "path": full_path,
                "installed": not downloading,
                "downloading": downloading,
                "disk_size_mb": disk_size_mb,
                "_rel_dir": rel_dir,
                "_filename": f,
            }
            if mmproj_files:
                entry["mmproj"] = os.path.join(root, mmproj_files[0])
            if record is not None:
                entry["download_metadata"] = record
            results.append(entry)

        # Download in progress but no main GGUF on disk yet: surface the
        # pending model from the marker so it still shows up in the listing.
        if marker is not None and not marker_listed:
            model_name = llamacpp_name_from_marker(marker, root)
            filename = gguf_basename(marker.get("model_url"))
            # Same naming as an installed file: declared downloads keep the marker
            # name (main() merges them into their entry), ad-hoc ones are qualified
            # by location so the id matches the one the finished install will get.
            if not any(declaration_covers(d, n, rel_dir, filename) for d, n, _mid in declarations):
                model_name = f"{rel_dir}/{model_name}" if rel_dir else model_name
            entry = {
                "id": f"llamacpp:{model_name}",
                "name": model_name,
                "handler": "llamacpp",
                "model_origin": ORIGIN_USER_CONFIGURED,
                "path": root,
                "installed": False,
                "downloading": True,
                "_rel_dir": rel_dir,
                "_filename": filename,
            }
            # A record for the file being fetched is a previous install of the same
            # model (a re-download); a record naming only other files is a sibling's.
            record = file_record(os.path.join(root, filename), llamacpp_dir) if filename else None
            if record is not None:
                entry["download_metadata"] = record
            results.append(entry)
    return results


def declared_model_id(declarations, rel_dir, filename, taken):
    """The id of the models-list.yaml entry declaring the GGUF at *rel_dir*/*filename*.

    A declaration matches at most one file per listing — *taken* holds the ids
    already matched — so a directory-level declaration cannot absorb every
    quantization sharing the repository directory. None when no entry declares the
    file: an ad-hoc download.
    """
    for directory, declared_name, model_id in declarations:
        if model_id in taken:
            continue
        if declaration_covers(directory, declared_name, rel_dir, filename):
            return model_id
    return None


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

    # Keep the platform of the board being listed (its variables drive the outdated check); the first one otherwise
    board = args.supported_board or os.environ.get("BOARD_NAME", "")
    deduped = {}
    for info in all_models:
        current = deduped.get(info["id"])
        if current is None or (board and info.get("platform") == board and current.get("platform") != board):
            deduped[info["id"]] = info
    all_models = list(deduped.values())

    results = []
    for model_info in all_models:
        if model_info.get("pre_loaded"):
            exists = True
            entry = {
                "id": model_info["id"],
                "name": model_info["name"],
                "handler": model_info["handler"],
                "model_origin": ORIGIN_BUILTIN,
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
                "model_origin": ORIGIN_BUILTIN,
                "installed": exists and not downloading,
                "downloading": downloading,
            }
            if model_info.get("model_size_mb") is not None:
                entry["model_size_mb"] = model_info["model_size_mb"]
            if exists:
                entry["path"] = path
                entry["disk_size_mb"] = get_dir_size_mb(path)
            # Read the record regardless of `exists`: check_model_exists() cannot
            # resolve the nested model_directory of a Hugging Face model, whose
            # status is only fixed up by the llamacpp merge below.
            metadata = model_metadata(model_info, args.models_dir, path if exists else None)
            if metadata is not None:
                entry["download_metadata"] = metadata
                stale = outdated_fields(model_info, metadata)
                entry["outdated"] = bool(stale)
                if stale:
                    entry["outdated_fields"] = stale

        results.append(entry)

    # Scan for llamacpp .gguf models on the filesystem. A scanned file may be the one
    # a models-list.yaml entry declares — matched by its location, since the entry's
    # nested model_directory the YAML path check couldn't resolve — and then its
    # filesystem status is merged into that entry instead of listing the model twice.
    # Anything else is an ad-hoc download and is listed as its own entry.
    by_id = {entry["id"]: entry for entry in results}
    info_by_id = {info["id"]: info for info in all_models}
    declarations = declared_gguf_files(models_list)
    merged_ids = set()
    for m in find_llamacpp_models(args.models_dir, declarations):
        rel_dir = m.pop("_rel_dir")
        filename = m.pop("_filename")
        declared_id = declared_model_id(declarations, rel_dir, filename, merged_ids)
        existing = by_id.get(declared_id) if declared_id else None
        if existing is not None:
            merged_ids.add(declared_id)
            # Keep the canonical YAML name/handler/model_size_mb/model_origin (the
            # model is declared, so it is not user-configured); take the
            # filesystem-derived status and on-disk details.
            existing["installed"] = m["installed"]
            existing["downloading"] = m["downloading"]
            if "path" in m:
                existing["path"] = m["path"]
            if m.get("disk_size_mb") is not None:
                existing["disk_size_mb"] = m["disk_size_mb"]
            if "mmproj" in m:
                existing["mmproj"] = m["mmproj"]
            if "download_metadata" in m and "download_metadata" not in existing:
                # The record was matched by the file's location rather than by the
                # entry's id — an ad-hoc install this catalog release adopted, whose
                # recorded id is the old path-qualified snapshot. Its inputs are
                # still what the install was downloaded with, so the outdated check
                # runs on them here, as it does for an id-matched record above.
                existing["download_metadata"] = m["download_metadata"]
                info = info_by_id.get(declared_id)
                if info is not None:
                    stale = outdated_fields(info, m["download_metadata"])
                    existing["outdated"] = bool(stale)
                    if stale:
                        existing["outdated_fields"] = stale
        else:
            if m["id"] in by_id:
                # Same name as a model this file is not: qualify the id by location
                qualified = f"{rel_dir}/{m['name']}" if rel_dir else m["name"]
                m["id"] = f"llamacpp:{qualified}"
                m["name"] = qualified
            results.append(m)
            by_id[m["id"]] = m

    # Apply installed/not-installed filters once, after merging.
    if args.installed_only:
        results = [r for r in results if r["installed"]]
    if args.not_installed_only:
        results = [r for r in results if not r["installed"]]

    if args.output_json:
        print(json.dumps({"event": "info", "models": results}, indent=2))
    else:
        installed_count = sum(1 for r in results if r["installed"])
        total_count = len(results)
        print(f"Models: {installed_count}/{total_count} installed\n")
        print(f"{'STATUS':<12} {'ORIGIN':<16} {'SIZE (MB)':<12} {'ID':<45} {'NAME':<40} {'PATH'}")
        print("-" * 169)
        for r in results:
            status = "DOWNLOADING" if r.get("downloading") else ("INSTALLED" if r["installed"] else "NOT FOUND")
            size = (
                f"{r['disk_size_mb']:.2f}"
                if r.get("disk_size_mb") is not None
                else (f"{r['model_size_mb']}" if r.get("model_size_mb") is not None else "-")
            )
            path = r.get("path", "")
            origin = r.get("model_origin", "")
            print(f"{status:<12} {origin:<16} {size:<12} {r['id']:<45} {r['name']:<40} {path}")


if __name__ == "__main__":
    main()
