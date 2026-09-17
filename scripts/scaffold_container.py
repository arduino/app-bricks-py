# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Scaffold a new container: its directory and Dockerfile, the docker-bake.hcl target, the
inventory row in containers/README.md and, unless --no-python says the image installs no Python
packages, the pyproject.toml with its license scan and Dependabot registrations.

  python3 -m scripts.scaffold_container my-runner --group ai --from python-slim --desc "What it runs"
  python3 -m scripts.scaffold_container my-runner --group ai --from python:3.13-slim-trixie@sha256:... --no-python

The group is one of the containers/<group>/ directories (ai, base, bricks), see containers/README.md.
The parent is either a container of this repository, linked in the Dockerfile and in the bake target
so it is built in-graph, or an external image reference. The Dockerfile and the dependency list are
starting points to edit, the printed next steps lock, scan and check the result.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.container_deps import ContainerDepsError, Containers  # noqa: E402

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
UV_IMAGE_PATTERN = re.compile(r"^FROM\s+(ghcr\.io/astral-sh/uv:\S+)\s+AS\s+uv\s*$", re.MULTILINE)
DEFAULT_UV_IMAGE = "ghcr.io/astral-sh/uv:0.10.3"

SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
"""


class ScaffoldError(RuntimeError):
    """Raised when the container cannot be scaffolded as requested."""


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
        raise ScaffoldError(f"docker-bake.hcl already has a target '{name}'.")
    blocks = target_blocks(hcl)
    if parent:
        missing = [target for target in subtree(containers, parent) if target not in blocks]
        if missing:
            raise ScaffoldError(f"docker-bake.hcl has no target for {', '.join(missing)}, run `task containers:check`.")
        anchor = max(subtree(containers, parent), key=lambda target: blocks[target][0])
    else:
        anchor = max(blocks, key=lambda target: blocks[target][1])
    end = blocks[anchor][1]
    hcl = hcl[:end] + "\n" + bake_target_text(name, group, parent) + hcl[end:]

    group = re.search(r'^group "default" \{\n  targets = \[\n(.*?)  \]\n', hcl, re.MULTILINE | re.DOTALL)
    if not group:
        raise ScaffoldError("docker-bake.hcl has no default group to list the target in.")
    entries = group.group(1)
    anchor_entry = f'    "{anchor}",\n'
    if anchor_entry not in entries:
        raise ScaffoldError(f"docker-bake.hcl lists no '{anchor}' in the default group.")
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
        raise ScaffoldError("containers/README.md has no inventory table.")
    anchor = next((r for r in rows if r.group(1) == parent), None) if parent else None
    insert_at = anchor.end() if anchor else rows[-1].end()
    readme = readme[:insert_at] + row + readme[insert_at:]

    graph = re.search(r"^```mermaid\n(.*?)^```\n", readme, re.MULTILINE | re.DOTALL)
    if not graph:
        raise ScaffoldError("containers/README.md has no mermaid hierarchy graph.")
    node = f"{mermaid_id(name)}[{name}]"
    if parent:
        parent_node = re.search(rf"(\w+)\[{re.escape(parent)}\]", graph.group(1))
        if not parent_node:
            raise ScaffoldError(f"The hierarchy graph in containers/README.md has no node for '{parent}'.")
        edge = f"  {parent_node.group(1)} --> {node}\n"
    else:
        edge = f"  {node}\n"
    return readme[: graph.end(1)] + edge + readme[graph.end(1) :]


def add_licensed_app(config: str, name: str, group: str) -> str:
    """Register the container's uv project in the dependency license scan."""
    if f"- name: {name}\n" in config:
        raise ScaffoldError(f".licensed.yml already has an app '{name}'.")
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
        raise ScaffoldError(".licensed.yml has no stale_records_action key to insert the app before.")
    return config.replace(marker, "\n" + app.rstrip("\n") + "\n" + marker, 1)


def add_dependabot_directory(config: str, name: str, group: str) -> str:
    """Add the container's uv project to the directories Dependabot updates."""
    entry = f"      - /containers/{group}/{name}\n"
    if entry in config:
        raise ScaffoldError(f".github/dependabot.yml already lists /containers/{group}/{name}.")
    entries = list(re.finditer(r"^      - /containers/[a-z0-9-]+/[a-z0-9-]+\n", config, re.MULTILINE))
    if not entries:
        raise ScaffoldError(".github/dependabot.yml lists no container uv project to insert after.")
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
        raise ScaffoldError(f"'{name}' is not a valid container name, use lowercase letters, digits and dashes.")
    containers_dir = repo_root / "containers"
    groups = sorted({path.parent.parent.name for path in containers_dir.glob("*/*/Dockerfile")})
    if group not in groups:
        raise ScaffoldError(f"'{group}' is not a container group, use one of {', '.join(groups)}.")
    directory = containers_dir / group / name
    if directory.exists():
        raise ScaffoldError(f"{directory} already exists.")
    containers = Containers(containers_dir)
    if name in containers.names:
        raise ScaffoldError(f"A container named '{name}' already exists.")

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
        steps.append(f"Declare the packages in containers/{group}/{name}/pyproject.toml, then run `task deps:lock` and `task license:deps`.")
    steps.append("Run `task containers:check` and `task containers:tree`, then build with `docker buildx bake " + name + "`.")
    return warnings + steps


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Container name, also the image name and the bake target.")
    parser.add_argument("--group", required=True, help="The containers/<group>/ directory to file it under: ai, base or bricks.")
    parser.add_argument("--from", dest="parent", required=True, metavar="PARENT", help="A container of this repo or an external image reference.")
    parser.add_argument("--desc", default="TODO", help="One line for the inventory in containers/README.md.")
    parser.add_argument(
        "--no-python",
        dest="python",
        action="store_false",
        help="The image installs no Python packages: skip pyproject.toml, license scan and Dependabot entries.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        steps = scaffold(args.repo_root, args.name, args.group, args.parent, args.desc, args.python)
    except (ScaffoldError, ContainerDepsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Scaffolded containers/{args.group}/{args.name}. Next steps:")
    for index, step in enumerate(steps, 1):
        print(f"  {index}. {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
