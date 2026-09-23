# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import ast
import os
from typing import Any
from dataclasses import dataclass, field
from docstring_parser import parse
import logging

logger = logging.getLogger(__name__)

MAX_INHERITANCE_DEPTH = 5  # base classes followed when collecting the members a brick inherits


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


def _import_map(tree: ast.AST, module_name: str) -> dict[str, tuple[str, str]]:
    """The names a module imports, as local name -> (module, name), relative imports resolved against the module."""
    package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module is not None or node.level):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(part for part in [*parts, base] if part)
            for alias in node.names:
                imports[alias.asname or alias.name] = (base, alias.name)
    return imports


def _module_file(module: str, file_path: str) -> str | None:
    """The source file of a module, looked up from the directories above the file being documented."""
    candidate = file_path
    for _ in range(10):
        candidate = os.path.dirname(candidate)
        for path in (os.path.join(candidate, *module.split("."), "__init__.py"), os.path.join(candidate, *module.split(".")) + ".py"):
            if os.path.isfile(path):
                return path
    logger.debug(f"Module {module} not found above {file_path}")
    return None


def _find_class(module: str, name: str, file_path: str, depth: int = 0) -> tuple[ast.ClassDef, str, str] | None:
    """The definition of a class, following the imports that re-export it: (class node, its module name, its file)."""
    path = _module_file(module, file_path)
    if path is None or depth > MAX_INHERITANCE_DEPTH:
        return None
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node, module, path
    imported = _import_map(tree, module + ".__init__" if path.endswith("__init__.py") else module).get(name)
    if imported is None:
        return None
    return _find_class(imported[0], imported[1], path, depth + 1)


def _inherited_members(class_node: ast.ClassDef, tree: ast.AST, module_name: str, file_path: str, depth: int = 0) -> tuple[list, list]:
    """The public methods and properties of the base classes of a class, the nearest base first, as documented
    in their own modules. Base classes outside the source tree are skipped."""
    methods, properties = [], []
    if depth > MAX_INHERITANCE_DEPTH:
        return methods, properties
    imports = _import_map(tree, module_name)
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            imported = imports.get(base.id, (module_name, base.id))
        elif isinstance(base, ast.Attribute):
            imported = (ast.unparse(base.value), base.attr)
        else:
            continue
        found = _find_class(imported[0], imported[1], file_path)
        if found is None:
            continue
        base_node, base_module, base_path = found
        with open(base_path, encoding="utf-8") as f:
            base_tree = ast.parse(f.read())
        base_methods, base_properties, _ = _class_members(base_node, module_name)
        deeper_methods, deeper_properties = _inherited_members(base_node, base_tree, base_module, base_path, depth + 1)
        methods += base_methods + deeper_methods
        properties += base_properties + deeper_properties
    return methods, properties


def _merge_members(own: list, inherited: list) -> list:
    """The members of a class: its own, then the inherited ones it does not redefine."""
    names = {member.name for member in own}
    merged = list(own)
    for member in inherited:
        if member.name not in names:
            names.add(member.name)
            merged.append(member)
    return merged


def _class_members(node: ast.ClassDef, module_name: str) -> tuple[list, list, list]:
    """The documented public methods and properties defined in a class body, and its __init__ parameters."""
    methods = []
    properties = []
    init_params = []
    property_setters = _get_property_setters(node)
    for stmt in node.body:
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
    return methods, properties, init_params


def extract_docstrings_with_types(file_path: str, module_name: str) -> list[DocstringInfo]:
    """Extract public class, method, and function docstrings and type hints from a Python file.

    The methods and properties a class inherits from base classes of the same source tree are documented
    with it, unless it redefines them.

    Args:
        file_path (str): Path to the Python file to analyze.
        module_name (str): Name of the module (used for dot notation in documentation).

    Returns:
        list[DocstringInfo]: A list of DocstringInfo objects describing classes, functions, and methods.
    """
    with open(file_path, encoding="utf-8") as f:
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
            for stmt in node.body:  # public attributes
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and not stmt.target.id.startswith("_"):
                    type_hints[stmt.target.id] = ast.unparse(stmt.annotation)
                    attrs.append((stmt.target.id, ast.unparse(stmt.annotation)))
            methods, properties, init_params = _class_members(node, module_name)
            inherited_methods, inherited_properties = _inherited_members(node, tree, module_name, file_path)
            methods = _merge_members(methods, inherited_methods)
            properties = _merge_members(properties, inherited_properties)
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
