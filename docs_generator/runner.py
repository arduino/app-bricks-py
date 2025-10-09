# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
from pathlib import Path
from docs_generator.extractor import extract_docstrings_with_types
from docs_generator.markdown_writer import generate_markdown
import logging
import ast

logger = logging.getLogger(__name__)


def process_app_bricks(src_root: str, output_dir: str):
    """Generate markdown API reference and example documentation for each brick in the app_bricks directory.

    If __all__ is present in __init__.py, only document the objects listed in __all__,
    otherwise document all public objects.
    """
    app_bricks_dir = os.path.join(src_root, "arduino", "app_bricks")
    logger.debug(f"Looking for app_bricks in: {app_bricks_dir}")
    if not os.path.exists(app_bricks_dir):
        logger.error(f"Directory not found: {app_bricks_dir}")
        return
    for folder in sorted(os.listdir(app_bricks_dir)):
        folder_path = os.path.join(app_bricks_dir, folder)
        logger.debug(f"Checking folder: {folder_path}")
        if os.path.isdir(folder_path):
            all_docstrings = []
            py_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".py")]
            logger.debug(f"Python files found: {py_files}")
            # Check for __all__ in __init__.py
            init_path = os.path.join(folder_path, "__init__.py")
            all_exports = None
            init_tree = None
            if os.path.exists(init_path):
                try:
                    with open(init_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)
                    init_tree = tree
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "__all__":
                                    try:
                                        all_exports = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.error(f"Error parsing __init__.py for __all__: {e}")
            # --- END NEW LOGIC ---
            docstrings_by_name = {}
            for file in py_files:
                try:
                    file_path = os.path.join(folder_path, file)
                    module_name = os.path.splitext(file)[0]
                    logger.info(f"Extracting docstrings from: {file_path} (module: {module_name})")
                    doc_items = extract_docstrings_with_types(file_path, module_name)
                    for item in doc_items:
                        # Select only classes/functions with unique names (overwrites in case of duplicates)
                        docstrings_by_name[item.name] = item
                except Exception as e:
                    logger.error(f"Error processing file {file}: {e}")
                    continue
            # If __all__ is present, filter only the names in __all__
            if all_exports is not None:
                # Default behaviour: include only names explicitly in __all__
                # For the streamlit_ui brick, try to also resolve names that are assigned/imported into exported objects
                if folder == "streamlit_ui":
                    expanded = []
                    # include direct matches first
                    for name in all_exports:
                        if name in docstrings_by_name and name not in expanded:
                            expanded.append(name)
                    # attempt to resolve assignments like "st.arduino_header = arduino_header"
                    if init_tree is not None:
                        for node in ast.walk(init_tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    # target like st.arduino_header
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                        exported_obj = target.value.id
                                        if exported_obj in all_exports:
                                            # RHS is a simple name
                                            if isinstance(node.value, ast.Name):
                                                src = node.value.id
                                                if src in docstrings_by_name and src not in expanded:
                                                    expanded.append(src)
                                            # RHS is an attribute (module.obj)
                                            elif isinstance(node.value, ast.Attribute):
                                                src = node.value.attr
                                                if src in docstrings_by_name and src not in expanded:
                                                    expanded.append(src)
                            # imports from .addons (e.g. "from .addons import arduino_header")
                            if isinstance(node, ast.ImportFrom):
                                mod = node.module or ""
                                if mod.endswith("addons") or mod.endswith(".addons"):
                                    for alias in node.names:
                                        name = alias.asname or alias.name
                                        if name in docstrings_by_name and name not in expanded:
                                            expanded.append(name)
                    all_docstrings = [docstrings_by_name[name] for name in expanded if name in docstrings_by_name]
                else:
                    all_docstrings = [docstrings_by_name[name] for name in all_exports if name in docstrings_by_name]
            else:
                all_docstrings = list(docstrings_by_name.values())
            # Create output folder for this brick
            brick_output_dir = os.path.join(output_dir, "arduino", "app_bricks", folder)
            os.makedirs(brick_output_dir, exist_ok=True)
            if all_docstrings:
                output_path = os.path.join(brick_output_dir, "API.md")
                logger.info(f"Generating markdown: {output_path}")
                generate_markdown(folder, all_docstrings, output_path)
            else:
                logger.info(f"No public docstrings found in folder: {folder_path}")


def process_app_peripherals(src_root: str, output_dir: str):
    """Generate markdown API reference and example documentation for each peripheral in the app_peripherals directory.

    If __all__ is present in __init__.py, only document the objects listed in __all__,
    otherwise document all public objects.
    """
    app_peripherals_dir = os.path.join(src_root, "arduino", "app_peripherals")
    logger.debug(f"Looking for app_peripherals in: {app_peripherals_dir}")
    if not os.path.exists(app_peripherals_dir):
        logger.error(f"Directory not found: {app_peripherals_dir}")
        return
    for folder in sorted(os.listdir(app_peripherals_dir)):
        folder_path = os.path.join(app_peripherals_dir, folder)
        logger.debug(f"Checking folder: {folder_path}")
        if os.path.isdir(folder_path):
            all_docstrings = []
            py_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".py")]
            logger.debug(f"Python files found: {py_files}")
            # Check for __all__ in __init__.py
            init_path = os.path.join(folder_path, "__init__.py")
            all_exports = None
            if os.path.exists(init_path):
                try:
                    with open(init_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "__all__":
                                    try:
                                        all_exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.error(f"Error parsing __init__.py for __all__: {e}")
            # --- END NEW LOGIC ---
            docstrings_by_name = {}
            for file in py_files:
                try:
                    file_path = os.path.join(folder_path, file)
                    module_name = os.path.splitext(file)[0]
                    logger.info(f"Extracting docstrings from: {file_path} (module: {module_name})")
                    doc_items = extract_docstrings_with_types(file_path, module_name)
                    for item in doc_items:
                        # Select only classes/functions with unique names (overwrites in case of duplicates)
                        docstrings_by_name[item.name] = item
                except Exception as e:
                    logger.error(f"Error processing file {file}: {e}")
                    continue
            # If __all__ is present, filter only the names in __all__
            if all_exports is not None:
                all_docstrings = [docstrings_by_name[name] for name in all_exports if name in docstrings_by_name]
            else:
                all_docstrings = list(docstrings_by_name.values())
            # Create output folder for this brick
            brick_output_dir = os.path.join(output_dir, "arduino", "app_peripherals", folder)
            os.makedirs(brick_output_dir, exist_ok=True)
            if all_docstrings:
                output_path = os.path.join(brick_output_dir, "API.md")
                logger.info(f"Generating markdown: {output_path}")
                generate_markdown(folder, all_docstrings, output_path)
            else:
                logger.info(f"No public docstrings found in folder: {folder_path}")


def process_app_peripherals(src_root: str, output_dir: str):
    """Generate markdown API reference and example documentation for each peripheral in the app_peripherals directory.

    If __all__ is present in __init__.py, only document the objects listed in __all__,
    otherwise document all public objects.
    """
    app_peripherals_dir = os.path.join(src_root, "arduino", "app_peripherals")
    logging.debug(f"Looking for app_peripherals in: {app_peripherals_dir}")
    if not os.path.exists(app_peripherals_dir):
        logging.error(f"Directory not found: {app_peripherals_dir}")
        return
    for folder in sorted(os.listdir(app_peripherals_dir)):
        folder_path = os.path.join(app_peripherals_dir, folder)
        logging.debug(f"Checking folder: {folder_path}")
        if os.path.isdir(folder_path):
            all_docstrings = []
            py_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".py")]
            logging.debug(f"Python files found: {py_files}")
            # Check for __all__ in __init__.py
            init_path = os.path.join(folder_path, "__init__.py")
            all_exports = None
            if os.path.exists(init_path):
                try:
                    with open(init_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "__all__":
                                    try:
                                        all_exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                                    except Exception:
                                        pass
                except Exception as e:
                    logging.error(f"Error parsing __init__.py for __all__: {e}")
            # --- END NEW LOGIC ---
            docstrings_by_name = {}
            for file in py_files:
                try:
                    file_path = os.path.join(folder_path, file)
                    module_name = os.path.splitext(file)[0]
                    logging.info(f"Extracting docstrings from: {file_path} (module: {module_name})")
                    doc_items = extract_docstrings_with_types(file_path, module_name)
                    for item in doc_items:
                        # Select only classes/functions with unique names (overwrites in case of duplicates)
                        docstrings_by_name[item.name] = item
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
                    continue
            # If __all__ is present, filter only the names in __all__
            if all_exports is not None:
                all_docstrings = [docstrings_by_name[name] for name in all_exports if name in docstrings_by_name]
            else:
                all_docstrings = list(docstrings_by_name.values())
            # Create output folder for this brick
            brick_output_dir = os.path.join(output_dir, "arduino", "app_peripherals", folder)
            os.makedirs(brick_output_dir, exist_ok=True)
            if all_docstrings:
                output_path = os.path.join(brick_output_dir, "API.md")
                logging.info(f"Generating markdown: {output_path}")
                generate_markdown(folder, all_docstrings, output_path)
            else:
                logging.info(f"No public docstrings found in folder: {folder_path}")


def run_docs_generator():
    """Entry point for the documentation generator CLI.

    Sets up the source and output directories, then generates API and example documentation for all bricks
    and peripherals.
    The output directory structure will mirror the source tree under 'docs/'.

    Returns:
        None
    """
    root_dir = Path(__file__).parent.parent
    source_root = root_dir / "src"
    output_directory = root_dir / "docs"
    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"Source root: {source_root}")
    logger.info(f"Output directory: {output_directory}")
    process_app_bricks(str(source_root), str(output_directory))
    process_app_peripherals(str(source_root), str(output_directory))


if __name__ == "__main__":
    run_docs_generator()
