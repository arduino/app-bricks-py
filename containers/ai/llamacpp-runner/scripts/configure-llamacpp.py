# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
from pathlib import Path

# The per-download record the models-downloader writes next to what it fetches.
# It is what tells an out-of-the-box model from a downloaded one, so the served
# names need no catalog baked into this image.
METADATA_NAME = ".arduino_metadata.yaml"


def downloaded_records(directory: Path):
    """The download records of *directory*'s ".arduino_metadata.yaml", newest last.

    The document is ``records: [...]``, one record per model downloaded into the
    directory (see the models-downloader's common/model_metadata.py). A missing or
    unusable file yields no records — the models there are then out-of-the-box.

    Mirrors the models-downloader's ``metadata_records``, keep the two in sync. This
    code is duplicated in the llamacpp-runner and llamacpp-npu-runner images.
    """
    try:
        import yaml

        with open(directory / METADATA_NAME) as f:
            data = yaml.safe_load(f)
    except Exception:
        return []
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def file_record(gguf_file: Path, models_dir: Path):
    """The download record describing *gguf_file*, or None when no record names it.

    The record lives in the directory the download landed in — for Hugging Face the
    repository directory, which can sit above a nested per-quantization folder — so
    every directory from the file's own up to *models_dir* is tried. Within a record,
    ``files`` holds paths relative to the record's directory; they are matched by
    full relative path or by basename, the two ways download patterns match a file.
    """
    directory = gguf_file.parent
    while True:
        rel = gguf_file.relative_to(directory).as_posix()
        for record in downloaded_records(directory):
            files = record.get("files")
            if isinstance(files, list) and any(isinstance(f, str) and (f == rel or f.split("/")[-1] == gguf_file.name) for f in files):
                return record
        if directory == models_dir or directory == directory.parent:
            return None
        directory = directory.parent


def gguf_model_name(gguf_file: Path, models_dir: Path) -> str:
    """The name llama-server serves this file under.

    Decided by the file's download record: a user-configured model (downloaded ad
    hoc) is named by its models_dir-relative path, so same-named files from different
    repositories never collide; a curated download (model_origin "builtin") keeps its
    file stem. A file with no record at all is an out-of-the-box model and keeps its
    stem too — that is the fallback, records only exist for downloaded models.

    The models-downloader derives the ``llamacpp:<name>`` ids of the same files, and
    the LLM brick resolves those against these sections, so the two namings may never
    drift apart. This code is duplicated in the llamacpp-runner and llamacpp-npu-runner
    images.
    """
    record = file_record(gguf_file, models_dir)
    if record is not None and record.get("model_origin") == "user_configured":
        return gguf_file.relative_to(models_dir).with_suffix("").as_posix()
    return gguf_file.stem


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        section = gguf_model_name(gguf_file, models_dir)
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
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    generate_models_ini(args.models_dir)
