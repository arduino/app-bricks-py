# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import configparser
import re
from pathlib import Path

# Parameter count as written in the model name: "4B", "0.8B", "E4B" -> 4.0, 0.8, 4.0.
# The lookahead skips version numbers ("gemma-4-", "Qwen3.5") and quantization tags ("Q4_0");
# \b cannot be used because "E4B_q4_0" has a word character right after the "B".
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z0-9.])", re.IGNORECASE)

# Models with at least this many billion parameters need 2 Hexagon sessions to run on the NPU.
BIG_MODEL_PARAMS_B = 4.0


def model_params_b(model_name: str):
    """Return the parameter count (in billions) read from a model name, or None if unknown."""
    matches = _PARAMS_RE.findall(model_name)
    if not matches:
        return None
    return max(float(m) for m in matches)


def detect_hexagon_ndev(model_names):
    """Return the number of Hexagon sessions required by the installed models."""
    ndev = 1
    for name in sorted(model_names):
        params = model_params_b(name)
        if params is None:
            print(f"  {name}: unknown parameter count, assuming it fits 1 session")
        elif params >= BIG_MODEL_PARAMS_B:
            print(f"  {name}: {params}B parameters, requires 2 sessions")
            ndev = 2
        else:
            print(f"  {name}: {params}B parameters, fits 1 session")

    return ndev


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()

    for gguf_file in sorted(models_dir.rglob("*.gguf")):
        if "mmproj" in gguf_file.name:
            continue

        section = gguf_file.stem
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

    return config.sections()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate models.ini from a models directory")
    parser.add_argument("models_dir", type=Path, help="Path to the models directory")
    parser.add_argument(
        "--ndev-out",
        type=Path,
        help="Write the number of Hexagon sessions required by the installed models to this file",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        raise SystemExit(f"Error: {args.models_dir} is not a directory")

    sections = generate_models_ini(args.models_dir)

    if args.ndev_out:
        print("Scanning installed models to size the Hexagon sessions...")
        ndev = detect_hexagon_ndev(sections)
        print(f"Recommended GGML_HEXAGON_NDEV={ndev}")

        # Not fatal: models.ini is already written and the caller falls back to the default.
        try:
            args.ndev_out.write_text(f"{ndev}\n")
        except OSError as e:
            print(f"Warning: could not write {args.ndev_out}: {e}")
