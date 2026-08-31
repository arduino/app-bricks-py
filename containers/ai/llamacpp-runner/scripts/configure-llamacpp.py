# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
from pathlib import Path

# The curated catalog, baked into the image, decides the served model names
MODELS_LIST_PATH = "/models-list.yaml"


def curated_declarations(models_list_path):
    """The (model_directory, filename) locations that models-list.yaml declares for llamacpp.
    A GGUF at a declared location is a curated model and keeps its file stem as the served
    name; every other file is ad-hoc and is named by its path.

    ``filename`` is None when the entry's model_url pins no file (a compact key naming
    a quantization); any GGUF inside the directory then counts as declared.

    An unreadable catalog degrades to no declarations: every model is then named by its path,
    and curated models fail instead of being served under an ambiguous stem.

    Mirrors the models-downloader's common/gguf_naming.py, keep the two in sync. This code
    is duplicated in the llamacpp-runner and llamacpp-npu-runner images.
    """
    try:
        import yaml

        with open(models_list_path) as f:
            entries = (yaml.safe_load(f) or {}).get("models", [])
    except Exception:
        return []

    declarations = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        for model_data in entry.values():
            if not isinstance(model_data, dict):
                continue
            deployment = model_data.get("deployment")
            platforms = deployment.get("platforms") if isinstance(deployment, dict) else None
            for platform_entry in platforms or []:
                if not isinstance(platform_entry, dict):
                    continue
                for platform_config in platform_entry.values():
                    variables = platform_config.get("variables") if isinstance(platform_config, dict) else None
                    if not isinstance(variables, dict):
                        continue
                    repository = str(variables.get("models_repository") or "")
                    if repository.rsplit("/models/", 1)[-1].removeprefix("models/") != "llamacpp":
                        continue
                    directory = str(variables.get("model_directory") or "").strip("/")
                    if not directory:
                        continue
                    url = str(variables.get("model_url") or "")
                    base = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
                    declaration = (directory, base if base.endswith(".gguf") else None)
                    if declaration not in declarations:
                        declarations.append(declaration)
    return declarations


def gguf_model_name(gguf_file: Path, models_dir: Path, declarations) -> str:
    """The name llama-server serves this file under (see curated_declarations)."""
    rel = gguf_file.relative_to(models_dir)
    rel_dir = rel.parent.as_posix()
    rel_dir = "" if rel_dir == "." else rel_dir
    for directory, filename in declarations:
        if rel_dir != directory and not rel_dir.startswith(directory + "/"):
            continue
        if filename is None or filename == gguf_file.name:
            return gguf_file.stem
    return rel.with_suffix("").as_posix()


def generate_models_ini(models_dir: Path, models_list_path: str):
    config = configparser.ConfigParser()
    declarations = curated_declarations(models_list_path)

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        section = gguf_model_name(gguf_file, models_dir, declarations)
        config[section] = {}
        config[section]["model"] = str(gguf_file.as_posix())

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            config[section]["mmproj"] = str(mmproj_files[0].as_posix())

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    print(f"Generated {output_path} with {len(config.sections())} model(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate models.ini from a models directory")
    parser.add_argument("models_dir", type=Path, help="Path to the models directory")
    parser.add_argument(
        "--models-list",
        default=MODELS_LIST_PATH,
        help=f"Path to the curated models-list.yaml (default: {MODELS_LIST_PATH})",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    generate_models_ini(args.models_dir, args.models_list)
