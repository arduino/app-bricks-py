# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The containers of the repository: their graph, derived from the Dockerfiles, and a scaffold for new ones.

Containers live in ``containers/<group>/<name>/``, the directory name being also the image name and
the group (ai, base, bricks) telling what the image is for. The base image of each one is declared
exactly once, in the ``FROM`` of its Dockerfile's final stage: this module resolves it through
multi-stage builds and tells whether it is another container of this repository
(``FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION}``) or an external image. CI needs no
second, drift-prone copy of the graph; ``docker-bake.hcl`` links the same parents so bake builds them
in order, and ``check-bake`` verifies the two agree.

    python3 -m scripts.container                  # JSON map of base image and parent per container
    python3 -m scripts.container list             # JSON array of every container
    python3 -m scripts.container closure NAME...  # the selection widened with its parents and children
    python3 -m scripts.container tree             # the hierarchy, grouped by external base image
    python3 -m scripts.container show NAME        # base image, parent, children and files of one container
    docker buildx bake --print | python3 -m scripts.container check-bake
    python3 -m scripts.container new my-runner --group ai --from python-slim --desc "What it runs"

``new`` creates the directory under its group with a starting Dockerfile, the bake target after its
parent, the inventory row in containers/README.md and, unless ``--no-python`` says the image installs
no Python packages, the pyproject.toml with its license scan and Dependabot registrations. ``--from``
is a container of this repository, built in-graph, or an external image reference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_GLOB = "*/*/Dockerfile"

FROM_PATTERN = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", re.IGNORECASE)
PARENT_PATTERN = re.compile(r"^\$\{REGISTRY\}app-bricks/([a-z0-9._-]+):\$\{BASE_IMAGE_VERSION\}$")


class ContainerError(RuntimeError):
    """Raised when the Dockerfiles do not describe a valid graph or a container cannot be scaffolded."""


def resolve_base_image(dockerfile: Path) -> str:
    """Return the image the Dockerfile's final stage builds on.

    Multi-stage builds produce the last stage, so resolution starts there and
    follows ``FROM <alias>`` references through earlier stages until it reaches
    an image that is not a stage of the same Dockerfile.
    """
    stages: list[tuple[str, str | None]] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = FROM_PATTERN.match(line)
        if match:
            stages.append((match.group(1), match.group(2)))
    if not stages:
        raise ContainerError(f"No FROM instruction in {dockerfile}")

    aliases = {alias.lower(): base for base, alias in stages if alias}
    base = stages[-1][0]
    seen: set[str] = set()
    while base.lower() in aliases:
        if base.lower() in seen:
            raise ContainerError(f"Circular stage references in {dockerfile}")
        seen.add(base.lower())
        base = aliases[base.lower()]
    return base


def parent_container(base_image: str) -> str | None:
    """Return the container name when the base image is built by this repository."""
    match = PARENT_PATTERN.match(base_image)
    return match.group(1) if match else None


class Containers:
    """The containers of the repository, with their base image and parent."""

    def __init__(self, containers_dir: Path) -> None:
        """Read every ``containers/<group>/<name>/Dockerfile``."""
        self.directory: dict[str, Path] = {}
        self.base: dict[str, str] = {}
        self.parent: dict[str, str | None] = {}

        dockerfiles = sorted(containers_dir.glob(DOCKERFILE_GLOB))
        if not dockerfiles:
            raise ContainerError(f"No containers found (looked for {DOCKERFILE_GLOB} under {containers_dir}).")

        for dockerfile in dockerfiles:
            name = dockerfile.parent.name
            if name in self.directory:
                raise ContainerError(
                    f"Duplicate container name '{name}': {self.directory[name]} and {dockerfile.parent}. "
                    f"Container names must be unique across groups (the name is also the image name)."
                )
            self.directory[name] = dockerfile.parent
            self.base[name] = resolve_base_image(dockerfile)
            self.parent[name] = parent_container(self.base[name])

        for name, parent in self.parent.items():
            if parent is not None and parent not in self.directory:
                raise ContainerError(f"'{name}' builds FROM unknown container '{parent}'.")

    @property
    def names(self) -> list[str]:
        """Every container name, sorted."""
        return sorted(self.directory)

    def children(self, name: str) -> list[str]:
        """The containers building FROM ``name``."""
        return sorted(child for child, parent in self.parent.items() if parent == name)

    def closure(self, selection: list[str]) -> list[str]:
        """Widen a selection so related images stay consistent.

        The containers deriving from the selection are added, so a parent is
        never rebuilt without its children, then the parents of the whole set,
        so every rebuilt image sits on a freshly built base.
        """
        unknown = sorted(set(selection) - set(self.directory))
        if unknown:
            raise ContainerError(f"Unknown container(s): {', '.join(unknown)}")

        selected = set(selection)
        frontier = set(selection)
        while frontier:
            frontier = {child for name in frontier for child in self.children(name)} - selected
            selected |= frontier
        frontier = set(selected)
        while frontier:
            frontier = {parent for name in frontier if (parent := self.parent[name])} - selected
            selected |= frontier
        return sorted(selected)

    def check_bake(self, definition: dict) -> list[str]:
        """Return what disagrees between a ``docker buildx bake --print`` definition and the Dockerfiles.

        Every container must be a target of the default group, every target must
        be a container, and a target must link exactly the parent its Dockerfile
        builds FROM: bake rewrites that FROM to the freshly built parent only
        through the link, without it the image is pulled from the registry instead.
        """
        targets = definition.get("target") or {}
        problems = [f"'{name}' has a Dockerfile but no bake target in the default group" for name in sorted(set(self.names) - set(targets))]
        problems += [f"bake target '{name}' has no Dockerfile under containers/" for name in sorted(set(targets) - set(self.names))]
        for name in sorted(set(self.names) & set(targets)):
            contexts = targets[name].get("contexts") or {}
            linked = sorted(value.removeprefix("target:") for value in contexts.values() if value.startswith("target:"))
            expected = [self.parent[name]] if self.parent[name] else []
            if linked != expected:
                problems.append(
                    f"bake target '{name}' links {', '.join(linked) or 'no parent'} but its Dockerfile builds FROM "
                    f"{expected[0] if expected else 'an external image'}"
                )
        return problems

    def describe(self, name: str) -> str:
        """Render what the repository knows about one container."""
        if name not in self.directory:
            raise ContainerError(f"Unknown container '{name}', see `list`.")
        directory = self.directory[name]
        lines = [
            f"{name}",
            f"  directory:  {directory}",
            f"  base image: {self.base[name]}",
            f"  parent:     {self.parent[name] or '-'}",
            f"  children:   {', '.join(self.children(name)) or '-'}",
            f"  python:     {'pyproject.toml + uv.lock' if (directory / 'pyproject.toml').exists() else '-'}",
            f"  tests:      {'tests/' if (directory / 'tests').is_dir() else '-'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, dict[str, str | None]]:
        """Map every container to its base image and parent container."""
        return {name: {"base": self.base[name], "parent": self.parent[name]} for name in self.names}

    def tree(self) -> str:
        """Render the hierarchy, grouped by external base image."""
        lines: list[str] = []

        def render(name: str, prefix: str) -> None:
            children = self.children(name)
            for index, child in enumerate(children):
                last = index == len(children) - 1
                lines.append(f"{prefix}{'└─' if last else '├─'} {child}")
                render(child, prefix + ("   " if last else "│  "))

        roots = [name for name in self.names if self.parent[name] is None]
        for base in sorted({self.base[root] for root in roots}):
            lines.append(base)
            base_roots = [root for root in roots if self.base[root] == base]
            for index, root in enumerate(base_roots):
                last = index == len(base_roots) - 1
                lines.append(f"{'└─' if last else '├─'} {root}")
                render(root, "   " if last else "│  ")
            lines.append("")
        return "\n".join(lines).rstrip()


NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
UV_IMAGE_PATTERN = re.compile(r"^FROM\s+(ghcr\.io/astral-sh/uv:\S+)\s+AS\s+uv\s*$", re.MULTILINE)
DEFAULT_UV_IMAGE = "ghcr.io/astral-sh/uv:0.10.3"

SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
"""


def dockerfile_text(name: str, base_image: str, parent: str | None, python: bool, uv_image: str) -> str:
    """The Dockerfile of a container building FROM ``base_image``, installing its locked packages when ``python``."""
    lines = [SPDX_HEADER]
    if parent:
        lines.append("ARG REGISTRY\nARG BASE_IMAGE_VERSION=latest\n")
    if python:
        lines.append(f"FROM {uv_image} AS uv\n")
    lines.append(f"FROM {base_image}\n")
    lines.append("ARG DEBIAN_FRONTEND=noninteractive\n")
    if python:
        lines.append(
            "COPY ./pyproject.toml ./uv.lock /tmp/deps/\n\n"
            "# uv is only mounted while installing, it is not part of the image\n"
            "RUN --mount=from=uv,source=/uv,target=/bin/uv \\\n"
            "    set -ex; \\\n"
            "    uv export --frozen --project /tmp/deps | uv pip install --system --no-cache-dir --require-hashes -r -; \\\n"
            "    rm -rf /tmp/deps; \\\n"
            "    # precompile python files to .pyc to speed up startup time\n"
            "    python -m compileall /usr/local/bin; \\\n"
            "    python -m compileall /usr/local/lib\n"
        )
    lines.append(f"# TODO: complete the image of {name}, see containers/README.md\n")
    if parent:
        lines.append("USER arduino\n")
    return "\n".join(lines)


def pyproject_text(name: str) -> str:
    """A pyproject.toml with no dependencies yet, locked for the boards but installable on developer machines."""
    return f"""# Python packages this image installs, pinned with hashes in uv.lock, see containers/README.md
[project]
name = "{name}"
version = "0"
requires-python = "==3.13.*"
dependencies = [
]

[tool.uv]
# The images run on aarch64 Linux, the lock must resolve for that target but installs on developer machines too
required-environments = ["sys_platform == 'linux' and platform_machine == 'aarch64'"]
"""


def bake_target_text(name: str, group: str, parent: str | None) -> str:
    """The docker-bake.hcl target of ``name``, linked to ``parent`` when it derives from a container of this repo."""
    inherits = "_downstream" if parent else "_common"
    lines = [
        f'target "{name}" {{',
        f'  inherits   = ["{inherits}"]',
        f'  context    = "containers/{group}/{name}"',
        f'  tags       = image_tags("{name}")',
        f'  cache-from = cache_from("{name}")',
        f'  cache-to   = cache_to("{name}")',
    ]
    if parent:
        lines.append(f'  contexts   = parent_context("{parent}")')
    lines.append("}")
    return "\n".join(lines) + "\n"


def target_blocks(hcl: str) -> dict[str, tuple[int, int]]:
    """Map every ``target "<name>"`` to the (start, end) offsets of its block."""
    blocks: dict[str, tuple[int, int]] = {}
    for match in re.finditer(r'^target "([^"]+)" \{\n.*?^\}\n', hcl, re.MULTILINE | re.DOTALL):
        blocks[match.group(1)] = (match.start(), match.end())
    return blocks


def subtree(containers: Containers, root: str) -> list[str]:
    """``root`` and every container deriving from it."""
    names = [root]
    for child in containers.children(root):
        names += subtree(containers, child)
    return names


def add_bake_target(hcl: str, name: str, group: str, parent: str | None, containers: Containers) -> str:
    """Insert the target after its parent's subtree and list it at the same spot of the default group."""
    if f'target "{name}"' in hcl:
        raise ContainerError(f"docker-bake.hcl already has a target '{name}'.")
    blocks = target_blocks(hcl)
    if parent:
        missing = [target for target in subtree(containers, parent) if target not in blocks]
        if missing:
            raise ContainerError(f"docker-bake.hcl has no target for {', '.join(missing)}, run `task check:containers:bake`.")
        anchor = max(subtree(containers, parent), key=lambda target: blocks[target][0])
    else:
        anchor = max(blocks, key=lambda target: blocks[target][1])
    end = blocks[anchor][1]
    hcl = hcl[:end] + "\n" + bake_target_text(name, group, parent) + hcl[end:]

    group = re.search(r'^group "default" \{\n  targets = \[\n(.*?)  \]\n', hcl, re.MULTILINE | re.DOTALL)
    if not group:
        raise ContainerError("docker-bake.hcl has no default group to list the target in.")
    entries = group.group(1)
    anchor_entry = f'    "{anchor}",\n'
    if anchor_entry not in entries:
        raise ContainerError(f"docker-bake.hcl lists no '{anchor}' in the default group.")
    entries = entries.replace(anchor_entry, anchor_entry + f'    "{name}",\n', 1)
    return hcl[: group.start(1)] + entries + hcl[group.end(1) :]


def mermaid_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name)


def add_inventory_row(readme: str, name: str, group: str, parent: str | None, base_image: str, desc: str) -> str:
    """Add the container to the inventory table, after its parent, and to the hierarchy graph."""
    built_from = f"`{parent}`" if parent else f"`{base_image.split('@')[0]}`"
    row = f"| `{name}` | {group} | {built_from} | {desc} |\n"
    rows = list(re.finditer(r"^\| `([^`]+)` \|.*\n", readme, re.MULTILINE))
    if not rows:
        raise ContainerError("containers/README.md has no inventory table.")
    anchor = next((r for r in rows if r.group(1) == parent), None) if parent else None
    insert_at = anchor.end() if anchor else rows[-1].end()
    readme = readme[:insert_at] + row + readme[insert_at:]

    graph = re.search(r"^```mermaid\n(.*?)^```\n", readme, re.MULTILINE | re.DOTALL)
    if not graph:
        raise ContainerError("containers/README.md has no mermaid hierarchy graph.")
    node = f"{mermaid_id(name)}[{name}]"
    if parent:
        parent_node = re.search(rf"(\w+)\[{re.escape(parent)}\]", graph.group(1))
        if not parent_node:
            raise ContainerError(f"The hierarchy graph in containers/README.md has no node for '{parent}'.")
        edge = f"  {parent_node.group(1)} --> {node}\n"
    else:
        edge = f"  {node}\n"
    return readme[: graph.end(1)] + edge + readme[graph.end(1) :]


def add_licensed_app(config: str, name: str, group: str) -> str:
    """Register the container's uv project in the dependency license scan."""
    if f"- name: {name}\n" in config:
        raise ContainerError(f".licensed.yml already has an app '{name}'.")
    app = f"""  - name: {name}
    source_path: .
    sources:
      pip: true
    python:
      virtual_env_dir: "/venvs/{name}"
    venv:
      project: containers/{group}/{name}

"""
    marker = "\nstale_records_action:"
    if marker not in config:
        raise ContainerError(".licensed.yml has no stale_records_action key to insert the app before.")
    return config.replace(marker, "\n" + app.rstrip("\n") + "\n" + marker, 1)


def add_dependabot_directory(config: str, name: str, group: str) -> str:
    """Add the container's uv project to the directories Dependabot updates."""
    entry = f"      - /containers/{group}/{name}\n"
    if entry in config:
        raise ContainerError(f".github/dependabot.yml already lists /containers/{group}/{name}.")
    entries = list(re.finditer(r"^      - /containers/[a-z0-9-]+/[a-z0-9-]+\n", config, re.MULTILINE))
    if not entries:
        raise ContainerError(".github/dependabot.yml lists no container uv project to insert after.")
    end = entries[-1].end()
    return config[:end] + entry + config[end:]


def detect_uv_image(containers_dir: Path) -> str:
    """The uv image the existing Dockerfiles mount, so every container installs with the same uv."""
    for dockerfile in sorted(containers_dir.glob("*/*/Dockerfile")):
        match = UV_IMAGE_PATTERN.search(dockerfile.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return DEFAULT_UV_IMAGE


def scaffold(repo_root: Path, name: str, group: str, parent_or_image: str, desc: str, python: bool) -> list[str]:
    """Create the container and register it everywhere the repository expects, returning the next steps."""
    if not NAME_PATTERN.match(name):
        raise ContainerError(f"'{name}' is not a valid container name, use lowercase letters, digits and dashes.")
    containers_dir = repo_root / "containers"
    groups = sorted({path.parent.parent.name for path in containers_dir.glob("*/*/Dockerfile")})
    if group not in groups:
        raise ContainerError(f"'{group}' is not a container group, use one of {', '.join(groups)}.")
    directory = containers_dir / group / name
    if directory.exists():
        raise ContainerError(f"{directory} already exists.")
    containers = Containers(containers_dir)
    if name in containers.names:
        raise ContainerError(f"A container named '{name}' already exists.")

    parent = parent_or_image if parent_or_image in containers.names else None
    base_image = f"${{REGISTRY}}app-bricks/{parent}:${{BASE_IMAGE_VERSION}}" if parent else parent_or_image
    warnings: list[str] = []
    if not parent and "@sha256:" not in base_image:
        warnings.append(f"'{base_image}' is not a container of this repo and carries no digest, pin it with @sha256:... in the Dockerfile.")

    bake = repo_root / "docker-bake.hcl"
    readme = containers_dir / "README.md"
    licensed = repo_root / ".licensed.yml"
    dependabot = repo_root / ".github" / "dependabot.yml"
    updates = {
        bake: add_bake_target(bake.read_text(encoding="utf-8"), name, group, parent, containers),
        readme: add_inventory_row(readme.read_text(encoding="utf-8"), name, group, parent, base_image, desc),
    }
    if python:
        updates[licensed] = add_licensed_app(licensed.read_text(encoding="utf-8"), name, group)
        updates[dependabot] = add_dependabot_directory(dependabot.read_text(encoding="utf-8"), name, group)

    directory.mkdir()
    (directory / "Dockerfile").write_text(dockerfile_text(name, base_image, parent, python, detect_uv_image(containers_dir)), encoding="utf-8")
    if python:
        (directory / "pyproject.toml").write_text(pyproject_text(name), encoding="utf-8")
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")

    steps = [f"Complete containers/{group}/{name}/Dockerfile and the purpose of '{name}' in containers/README.md."]
    if python:
        steps.append(f"Declare the packages in containers/{group}/{name}/pyproject.toml, then run `task deps:lock` and `task fix:licenses`.")
    steps.append(f"Run `task show:containers`, then build with `task build:containers -- {name}`.")
    return warnings + steps


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="Print every container name as a JSON array.")
    closure_parser = subparsers.add_parser("closure", help="Widen a selection with its parents and children, as a JSON array.")
    closure_parser.add_argument("containers", nargs="+", help="Selected container names.")
    subparsers.add_parser("tree", help="Print the container hierarchy.")
    show_parser = subparsers.add_parser("show", help="Print the base image, parent, children and files of one container.")
    show_parser.add_argument("name")
    subparsers.add_parser("check-bake", help="Check a `docker buildx bake --print` definition, read from stdin, against the Dockerfiles.")
    new_parser = subparsers.add_parser("new", help="Scaffold a container and register it everywhere the repository expects.")
    new_parser.add_argument("name", help="Container name, also the image name and the bake target.")
    new_parser.add_argument("--group", required=True, help="The containers/<group>/ directory to file it under: ai, base or bricks.")
    new_parser.add_argument("--from", dest="parent", required=True, metavar="PARENT", help="A container of this repo or an external image reference.")
    new_parser.add_argument("--desc", default="TODO", help="One line for the inventory in containers/README.md.")
    new_parser.add_argument(
        "--no-python",
        dest="python",
        action="store_false",
        help="The image installs no Python packages: skip pyproject.toml, license scan and Dependabot entries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "new":
            steps = scaffold(args.repo_root, args.name, args.group, args.parent, args.desc, args.python)
            print(f"Scaffolded containers/{args.group}/{args.name}. Next steps:")
            for index, step in enumerate(steps, 1):
                print(f"  {index}. {step}")
            return 0
        containers = Containers(args.repo_root / "containers")
        if args.command == "list":
            print(json.dumps(containers.names))
        elif args.command == "closure":
            print(json.dumps(containers.closure(args.containers)))
        elif args.command == "tree":
            print(containers.tree())
        elif args.command == "show":
            print(containers.describe(args.name))
        elif args.command == "check-bake":
            problems = containers.check_bake(json.load(sys.stdin))
            for problem in problems:
                print(f"Error: {problem}", file=sys.stderr)
            if problems:
                return 1
            print("docker-bake.hcl agrees with the Dockerfiles")
        else:
            print(json.dumps(containers.to_dict(), indent=2))
    except ContainerError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
