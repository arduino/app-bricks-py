# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from docs_generator.extractor import DocstringInfo
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


def _log_docstring_item(item: DocstringInfo):
    logger.debug(f"Processing item: {item.name} ({item.kind})")
    logger.debug(f"{item.name} - Blank_after_long_description: {getattr(item.doc, 'blank_after_long_description', None)}")
    logger.debug(f"{item.name} - blank_after_short_description: {getattr(item.doc, 'blank_after_short_description', None)}")
    logger.debug(f"{item.name} - long_description: {getattr(item.doc, 'long_description', None)}")
    logger.debug(f"{item.name} - meta: {getattr(item.doc, 'meta', None)}")
    for meta in getattr(item.doc, "meta", []):
        logger.debug(f"{item.name} - meta.args: {getattr(meta, 'args', None)}")
        logger.debug(f"{item.name} - meta.description: {getattr(meta, 'description', None)}")
    logger.debug(f"{item.name} - short_description: {getattr(item.doc, 'short_description', None)}")
    logger.debug(f"{item.name} - Style: {getattr(item.doc, 'style', None)}")
    logger.debug(f"{item.name} - Deprecation: {getattr(item.doc, 'deprecation', None)}")
    logger.debug(f"{item.name} - Examples: {getattr(item.doc, 'examples', None)}")
    for example in getattr(item.doc, "examples", []):
        logger.debug(f"{item.name} - example.snippet: {getattr(example, 'snippet', None)}")
        logger.debug(f"{item.name} - example.description: {getattr(example, 'description', None)}")
    logger.debug(f"{item.name} - many_returns: {getattr(item.doc, 'many_returns', None)}")
    logger.debug(f"{item.name} - params: {getattr(item.doc, 'params', None)}")
    for param in getattr(item.doc, "params", []):
        logger.debug(f"{item.name} - param.arg_name: {getattr(param, 'arg_name', None)}")
        logger.debug(f"{item.name} - param.default: {getattr(param, 'default', None)}")
        logger.debug(f"{item.name} - param.is_optional: {getattr(param, 'is_optional', None)}")
        logger.debug(f"{item.name} - param.type_name: {getattr(param, 'type_name', None)}")
        logger.debug(f"{item.name} - param.args: {getattr(param, 'args', None)}")
        logger.debug(f"{item.name} - param.description: {getattr(param, 'description', None)}")
    logger.debug(f"{item.name} - Raises: {getattr(item.doc, 'raises', None)}")
    logger.debug(f"{item.name} - Returns: {getattr(item.doc, 'returns', None)}")


def _format_parameters(params: List[Any], heading_level: int = 4) -> str:
    if not params:
        return ""
    out = f"{'#' * heading_level} Parameters\n\n"
    for param in params:
        type_str = f" (*{param.type_name}*)" if param.type_name else ""
        default_str = f", default={param.default}" if param.default is not None else ""
        optional_str = " (optional)" if getattr(param, "is_optional", False) else ""
        out += f"- **{param.arg_name}**{type_str}{optional_str}{default_str}: {param.description}\n"
    return out + "\n"


def _format_returns(returns: Any, heading_level: int = 4) -> str:
    if not returns:
        return ""
    type_str = f" (*{returns.type_name}*)" if getattr(returns, "type_name", None) else ""
    desc = getattr(returns, "description", None) or ""
    return f"{'#' * heading_level} Returns\n\n-{type_str}: {desc}\n\n"


def _format_raises(raises: List[Any], heading_level: int = 4) -> str:
    if not raises:
        return ""
    out = f"{'#' * heading_level} Raises\n\n"
    for exc in raises:
        type_str = f"**{getattr(exc, 'type_name', '')}**" if getattr(exc, "type_name", None) else ""
        desc = getattr(exc, "description", None) or ""
        out += f"- {type_str}: {desc}\n"
    return out + "\n"


def _format_examples(examples: List[Any], heading_level: int = 3) -> str:
    if not examples:
        return ""
    out = f"{'#' * heading_level} Examples\n\n"
    for ex in examples:
        desc = getattr(ex, "description", None) or ""
        snippet = getattr(ex, "snippet", None)
        if desc and not snippet:
            out += f"```python\n{desc}\n```\n"
        elif desc and snippet:
            out += f"{desc}\n"
            out += f"```python\n{snippet}\n```\n"
        elif snippet:
            out += f"```python\n{snippet}\n```\n"
    return out


def generate_markdown(folder_name: str, docstrings: list[DocstringInfo], output_path: str):
    """Generates a markdown file for a folder, including all public API docstrings and type hints.

    Args:
        folder_name (str): The name of the folder/module being documented.
        docstrings (list[DocstringInfo]):
            List of extracted docstring and type information for classes, functions, and methods.
        output_path (str): Path where the generated markdown file will be saved.

    Returns:
        None
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {folder_name} API Reference\n\n")
        # Collect section titles for index (level II headings only, no links)
        section_titles = []
        for item in docstrings:
            if item.kind == "class":
                section_titles.append(f"- Class `{item.name}`")
            elif item.kind == "function":
                dot_name = f"{item.module_name}.{item.name}" if item.module_name != "__init__" else item.name
                section_titles.append(f"- Function `{dot_name}`")
        if section_titles:
            f.write("## Index\n\n")
            for line in section_titles:
                f.write(line + "\n")
            f.write("\n---\n\n")
        for idx, item in enumerate(docstrings):
            _log_docstring_item(item)
            # Add horizontal rule between objects (not before the first)
            if idx > 0:
                f.write("\n---\n\n")
            # Title and signature
            if item.kind == "class":
                # Safe check: parsed docstring may be None
                is_dataclass = getattr(item.doc, "short_description", None) and "data class" in item.doc.short_description.lower()
                class_title = f"## `{item.name}` dataclass\n\n" if is_dataclass else f"## `{item.name}` class\n\n"
                f.write(class_title)
                # Signature chunk: class ...
                f.write(f"""```python\nclass {item.signature}\n```\n\n""")
                # Descriptions
                if getattr(item.doc, "short_description", None):
                    f.write(f"{item.doc.short_description}\n\n")
                if getattr(item.doc, "long_description", None):
                    f.write(f"{item.doc.long_description}\n\n")
                # Attributes (from params with 'attribute' in args)
                attrs = [p for p in getattr(item.doc, "params", []) if "attribute" in getattr(p, "args", [])]
                # For dataclass: show only attributes
                if is_dataclass:
                    if attrs:
                        f.write("### Attributes\n\n")
                        for attr in attrs:
                            type_str = f" (*{attr.type_name}*)" if attr.type_name else ""
                            f.write(f"- **{attr.arg_name}**{type_str}: {attr.description}\n")
                        f.write("\n")
                else:
                    # For normal classes: show Parameters (all params that are not attributes)
                    # Find __init__ method in item.methods
                    init_method = None
                    other_methods = []
                    for m in item.methods:
                        if m.name == "__init__":
                            init_method = m
                        else:
                            other_methods.append(m)
                    if init_method and getattr(init_method.doc, "params", None):
                        f.write(_format_parameters(init_method.doc.params, heading_level=3))
                        f.write(_format_returns(init_method.doc.returns, heading_level=3))
                        f.write(_format_raises(init_method.doc.raises, heading_level=3))
                        f.write(_format_examples(init_method.doc.examples, heading_level=3))
                    if attrs:
                        f.write("### Attributes\n\n")
                        for attr in attrs:
                            type_str = f" (*{attr.type_name}*)" if attr.type_name else ""
                            f.write(f"- **{attr.arg_name}**{type_str}: {attr.description}\n")
                        f.write("\n")
                # Methods (skip __init__)
                if not is_dataclass and other_methods:
                    f.write("### Methods\n\n")
                    for m in other_methods:
                        f.write(f"#### `{m.signature}`\n\n")
                        if m.doc.short_description:
                            f.write(f"{m.doc.short_description}\n\n")
                        if m.doc.long_description:
                            f.write(f"{m.doc.long_description}\n\n")
                        f.write(_format_parameters(m.doc.params, heading_level=5))
                        f.write(_format_returns(m.doc.returns, heading_level=5))
                        f.write(_format_raises(m.doc.raises, heading_level=5))
                        f.write(_format_examples(m.doc.examples, heading_level=5))
                elif is_dataclass and item.methods:
                    f.write("### Methods\n\n")
                    for m in item.methods:
                        f.write(f"#### `{m.signature}`\n\n")
                        if m.doc.short_description:
                            f.write(f"{m.doc.short_description}\n\n")
                        if m.doc.long_description:
                            f.write(f"{m.doc.long_description}\n\n")
                        f.write(_format_parameters(m.doc.params, heading_level=5))
                        f.write(_format_returns(m.doc.returns, heading_level=5))
                        f.write(_format_raises(m.doc.raises, heading_level=5))
                        f.write(_format_examples(m.doc.examples, heading_level=5))
            elif item.kind == "function":
                dot_name = f"{item.module_name}.{item.name}" if item.module_name != "__init__" else item.name
                is_decorator = False
                if item.name and item.name[0].islower() and item.doc.short_description:
                    if "decorator" in item.doc.short_description.lower():
                        is_decorator = True
                if is_decorator:
                    f.write(f"## `{dot_name}` function decorator\n\n")
                    # Signature chunk: @decorator ...
                    f.write(f"""```python\n@{item.signature}\n```\n\n""")
                else:
                    f.write(f"## `{dot_name}` function\n\n")
                    # Signature chunk: def ...
                    f.write(f"""```python\ndef {item.signature}\n```\n\n""")
                if item.doc.short_description:
                    f.write(f"{item.doc.short_description}\n\n")
                if item.doc.long_description:
                    f.write(f"{item.doc.long_description}\n\n")
                f.write(_format_parameters(item.doc.params, heading_level=3))
                f.write(_format_returns(item.doc.returns, heading_level=3))
                f.write(_format_raises(item.doc.raises, heading_level=3))
                f.write(_format_examples(item.doc.examples, heading_level=3))
