# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import ast
from typing import Any
from dataclasses import dataclass, field
from docstring_parser import parse
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocstringInfo:
    """Container for extracted docstring and type information for a class, function, or method.

    Attributes:
        kind (str): 'class', 'function', 'method', or 'property'.
        name (str): Name of the class, function, or method.
        signature (str): Formatted signature string.
        doc (Any): Parsed docstring object.
        type_hints (dict[str, str]): Dictionary mapping argument/attribute names to types.
        module_name (str): Module name for dot notation.
        methods (list): For classes, a list of DocstringInfo for methods; empty for functions.
        properties (list): For classes, a list of DocstringInfo for properties; empty otherwise.
        is_readonly (bool): True when the property does not expose a setter.
    """

    kind: str  # 'class', 'function', 'method', or 'property'
    name: str
    signature: str
    doc: Any
    type_hints: dict[str, str]
    module_name: str
    methods: list["DocstringInfo"] = field(default_factory=list)
    properties: list["DocstringInfo"] = field(default_factory=list)
    is_readonly: bool = False


def _extract_all_exports(tree: ast.AST) -> list[str] | None:
    all_exports = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        all_exports = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_exports.append(elt.value)
                    except Exception:
                        pass
    return all_exports


def _get_property_setters(class_node: ast.ClassDef) -> set[str]:
    setter_names = set()
    for stmt in class_node.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for decorator in stmt.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == "setter" and isinstance(decorator.value, ast.Name):
                setter_names.add(decorator.value.id)
    return setter_names


def _is_property_getter(function_node: ast.FunctionDef) -> bool:
    return any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in function_node.decorator_list)


def _is_property_accessor(function_node: ast.FunctionDef) -> bool:
    return any(isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"} for decorator in function_node.decorator_list)


def extract_docstrings_with_types(file_path: str, module_name: str) -> list[DocstringInfo]:
    """Extract public class, method, and function docstrings and type hints from a Python file.

    Args:
        file_path (str): Path to the Python file to analyze.
        module_name (str): Name of the module (used for dot notation in documentation).

    Returns:
        list[DocstringInfo]: A list of DocstringInfo objects describing classes, functions, and methods.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    # Patch AST nodes to know their parent for top-level function detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    docstrings = []
    # Parse __all__ if present
    all_exports = _extract_all_exports(tree)
    for node in ast.walk(tree):
        # Only public classes and functions
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            if all_exports is not None and node.name not in all_exports:
                continue
            docstring = ast.get_docstring(node)
            if docstring is not None:
                parsed = parse(docstring)
            else:
                parsed = None
            type_hints = {}
            attrs = []
            methods = []
            properties = []
            init_params = []
            property_setters = _get_property_setters(node)
            # Look for __init__ to extract constructor parameters
            for stmt in node.body:
                # Public attributes
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and not stmt.target.id.startswith("_"):
                    type_hints[stmt.target.id] = ast.unparse(stmt.annotation)
                    attrs.append((stmt.target.id, ast.unparse(stmt.annotation)))
                # Public methods OR __init__
                if isinstance(stmt, ast.FunctionDef) and (not stmt.name.startswith("_") or stmt.name == "__init__"):
                    if _is_property_getter(stmt):
                        p_docstring = ast.get_docstring(stmt)
                        if p_docstring:
                            p_return_type = ast.unparse(stmt.returns) if stmt.returns else ""
                            p_signature = f"{stmt.name}: {p_return_type}" if p_return_type else stmt.name
                            properties.append(
                                DocstringInfo(
                                    kind="property",
                                    name=stmt.name,
                                    signature=p_signature,
                                    doc=parse(p_docstring),
                                    type_hints={},
                                    module_name=module_name,
                                    is_readonly=stmt.name not in property_setters,
                                )
                            )
                        continue
                    if _is_property_accessor(stmt):
                        continue
                    m_docstring = ast.get_docstring(stmt)
                    if m_docstring:
                        m_parsed = parse(m_docstring)
                        m_type_hints = {}
                        m_args = []
                        for arg in stmt.args.args:
                            if arg.arg == "self" or arg.arg.startswith("_"):
                                continue
                            t = ast.unparse(arg.annotation) if arg.annotation else ""
                            m_type_hints[arg.arg] = t
                            m_args.append((arg.arg, t))
                        m_sig = f"{stmt.name}({', '.join(f'{a[0]}: {a[1]}' if a[1] else a[0] for a in m_args)})"
                        methods.append(
                            DocstringInfo(
                                kind="method",
                                name=stmt.name,
                                signature=m_sig,
                                doc=m_parsed,
                                type_hints=m_type_hints,
                                module_name=module_name,
                            )
                        )
                    # If __init__, save params
                    if stmt.name == "__init__":
                        init_params.clear()
                        for arg in stmt.args.args:
                            if arg.arg == "self" or arg.arg.startswith("_"):
                                continue
                            t = ast.unparse(arg.annotation) if arg.annotation else ""
                            init_params.append((arg.arg, t))
            # Class signature: if dataclass use attributes, else use __init__ params (with type)
            is_dataclass = any(d.id == "dataclass" if isinstance(d, ast.Name) else False for d in getattr(node, "decorator_list", []))
            # DEBUG: Log class name, init_params, attrs
            logger.debug(f"Class: {node.name}, is_dataclass: {is_dataclass},")
            logger.debug(f"init_params: {init_params},")
            logger.debug(f"attrs: {attrs}")
            if is_dataclass:
                sig = f"{node.name}({', '.join(f'{a[0]}: {a[1]}' for a in attrs)})"
            elif init_params:
                sig = f"{node.name}(" + ", ".join(f"{a[0]}: {a[1]}" if a[1] else a[0] for a in init_params) + ")"
            else:
                sig = f"{node.name}()"
            logger.debug(f"Class: {node.name}, signature: {sig}")
            # Merge __init__ params into attributes if not already present
            for p, t in init_params:
                if p not in type_hints:
                    type_hints[p] = t
                    attrs.append((p, t))
            docstrings.append(
                DocstringInfo(
                    kind="class",
                    name=node.name,
                    signature=sig,
                    doc=parsed,
                    type_hints=type_hints,
                    module_name=module_name,
                    methods=methods,
                    properties=properties,
                )
            )
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and isinstance(node.parent, ast.Module):
            if all_exports is not None and node.name not in all_exports:
                continue
            docstring = ast.get_docstring(node)
            if docstring:
                parsed = parse(docstring)
                type_hints = {}
                args = []
                for arg in node.args.args:
                    if arg.arg == "self" or arg.arg.startswith("_"):
                        continue
                    t = ast.unparse(arg.annotation) if arg.annotation else ""
                    type_hints[arg.arg] = t
                    args.append((arg.arg, t))
                # Function signature: def func(arg1: type1, ...)
                sig = f"{node.name}({', '.join(f'{a[0]}: {a[1]}' if a[1] else a[0] for a in args)})"
                docstrings.append(
                    DocstringInfo(
                        kind="function",
                        name=node.name,
                        signature=sig,
                        doc=parsed,
                        type_hints=type_hints,
                        module_name=module_name,
                    )
                )
    return docstrings
