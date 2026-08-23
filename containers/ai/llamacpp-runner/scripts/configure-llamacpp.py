# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
from collections import Counter
from pathlib import Path


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    stems = Counter(p.stem for p in gguf_files)
    for gguf_file in gguf_files:
        if stems[gguf_file.stem] == 1:
            section = gguf_file.stem
        else:
            section = gguf_file.relative_to(models_dir).with_suffix("").as_posix()
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
