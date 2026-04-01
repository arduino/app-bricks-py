# Docs generator

A simple, robust Python documentation generator for producing clean, PEP8-compliant, Google-style Markdown documentation for Arduino Bricks.

## Features

- Extracts and documents modules, classes, functions, methods, and class properties, including constructor (`__init__`) parameters.
- Filters documented objects in `__init__.py` files to only those listed in `__all__`, if present.
- Mirrors the source folder structure under `docs/`, organizing documentation by module.
- Produces readable Markdown with clear sections: Parameters, Returns, Raises, and Examples.
- Renders class properties in a dedicated `Properties` section instead of mixing them with methods.
- Generates an index at the top of each file listing all documented objects.
- No HTML anchors or non-standard Markdown.
- Uses logging for debugging and robust error handling.

## Usage

1. Run the documentation generator:

   ```sh
   python -m docs_generator.runner
   ```

2. Find the generated documentation under the `docs/` directory, mirroring arduino source structure.

## Structure

- `extractor.py`: Extracts docstrings, type hints, and public API info.
- `markdown_writer.py`: Formats and writes Markdown documentation, including dedicated sections for properties.
- `runner.py`: Orchestrates the process and mirrors the folder structure.

## Requirements

- Python 3.13+
- docstring_parser 0.16+
