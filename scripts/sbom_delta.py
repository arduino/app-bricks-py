#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Compute delta reports between container SBOMs.

The SBOM of a published container is the SPDX attestation BuildKit attached to
it at build time, read with `docker buildx imagetools inspect`. The delta is
computed against the base image of its Dockerfile: a parent container of this
repository, whose attestation is read the same way, or an external image, the
only kind still scanned with Syft.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
REPO_ROOT = Path(__file__).resolve().parents[1]

# Lets the script run both as `python3 -m scripts.sbom_delta` and directly
sys.path.insert(0, str(REPO_ROOT))
from scripts.container_deps import ContainerDepsError, Containers  # noqa: E402


class SbomDeltaError(RuntimeError):
    """Raised when SBOM delta input or configuration is invalid."""


def normalize_registry(registry: str) -> str:
    """Ensure the registry prefix ends with a slash when not empty."""
    if not registry or registry.endswith("/"):
        return registry
    return f"{registry}/"


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON file as a dict."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SbomDeltaError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SbomDeltaError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SbomDeltaError(f"Expected JSON object in {path}.")
    return payload


def resolve_template(template: str, context: dict[str, str]) -> str:
    """Resolve ``${VAR}`` placeholders inside a template string."""
    resolved = template
    for _ in range(10):
        matches = VARIABLE_PATTERN.findall(resolved)
        if not matches:
            return resolved
        missing = sorted({name for name in matches if name not in context})
        if missing:
            raise SbomDeltaError(f"Missing template variables: {', '.join(missing)}")
        resolved = VARIABLE_PATTERN.sub(lambda match: context[match.group(1)], resolved)
    raise SbomDeltaError(f"Could not resolve template after 10 passes: {template}")


def resolve_runtime_base(containers: Containers, container: str, registry: str, version: str) -> str:
    """Resolve the image the delta SBOM is computed against.

    It is the base of the container's Dockerfile: a parent container of this
    repository at the given registry and version, or an external image as written.
    """
    if container not in containers.base:
        raise SbomDeltaError(f"Unknown container '{container}'. Known: {', '.join(containers.names)}.")
    context = {"REGISTRY": normalize_registry(registry), "BASE_IMAGE_VERSION": version}
    return resolve_template(containers.base[container], context)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Package:
    """Lightweight representation of a software package from a Syft SBOM."""

    __slots__ = ("name", "version", "pkg_type", "purl", "licenses")

    def __init__(
        self,
        name: str,
        version: str,
        pkg_type: str,
        purl: str | None,
        licenses: list[str],
    ) -> None:
        self.name = name
        self.version = version
        self.pkg_type = pkg_type
        self.purl = purl
        self.licenses = licenses


# (lowercase name, lowercase type) -> Package
PkgMap = dict[tuple[str, str], Package]
# (lowercase name, lowercase type) -> (old Package, new Package)
UpdateMap = dict[tuple[str, str], tuple[Package, Package]]

# Package types that represent image metadata, not real software dependencies.
_IGNORED_PACKAGE_TYPES = frozenset({"oci"})


# ---------------------------------------------------------------------------
# Parsing (Syft native JSON)
# ---------------------------------------------------------------------------


def _extract_license(lic_declared: str) -> str:
    """Return the license string if meaningful, empty string otherwise."""
    if lic_declared and lic_declared != "NOASSERTION" and lic_declared != "NONE":
        return lic_declared
    return ""


def _extract_type_from_purl(purl: str) -> str:
    """Extract the package type (e.g. 'pypi', 'deb') from a purl."""
    match = re.match(r"pkg:([^/]+)/", purl)
    return match.group(1) if match else "unknown"


def load_spdx_packages(sbom_path: Path) -> PkgMap:
    """Load all packages from an SPDX 2.3 JSON SBOM."""
    sbom = load_json(sbom_path)
    packages: PkgMap = {}
    for pkg in sbom.get("packages", []):
        if not isinstance(pkg, dict):
            continue
        spdxid = pkg.get("SPDXID", "")
        if spdxid == "SPDXRef-DOCUMENT":
            continue

        name: str = pkg.get("name", "")
        version: str = pkg.get("versionInfo", "")

        # Extract purl and type from externalRefs
        purl: str | None = None
        pkg_type = "unknown"
        for ref in pkg.get("externalRefs") or []:
            if isinstance(ref, dict) and ref.get("referenceType") == "purl":
                purl = str(ref["referenceLocator"])
                pkg_type = _extract_type_from_purl(purl)
                break

        if pkg_type.lower() in _IGNORED_PACKAGE_TYPES:
            continue

        # Extract license
        lic_str = _extract_license(pkg.get("licenseDeclared", ""))
        licenses: list[str] = [lic_str] if lic_str else []

        key = (name.lower(), pkg_type.lower())
        packages[key] = Package(
            name=name,
            version=version,
            pkg_type=pkg_type,
            purl=purl,
            licenses=licenses,
        )
    return packages


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_delta(
    base_sbom_path: Path,
    full_sbom_path: Path,
) -> tuple[PkgMap, UpdateMap, PkgMap]:
    """Return (added, updated, removed) package maps."""
    base_pkgs = load_spdx_packages(base_sbom_path)
    child_pkgs = load_spdx_packages(full_sbom_path)

    added: PkgMap = {}
    updated: UpdateMap = {}
    removed: PkgMap = {}

    for key, pkg in child_pkgs.items():
        if key not in base_pkgs:
            added[key] = pkg
        elif base_pkgs[key].version != pkg.version:
            updated[key] = (base_pkgs[key], pkg)

    for key, pkg in base_pkgs.items():
        if key not in child_pkgs:
            removed[key] = pkg

    return added, updated, removed


# ---------------------------------------------------------------------------
# SPDX 2.3 output
# ---------------------------------------------------------------------------

_SPDXID_RE = re.compile(r"[^A-Za-z0-9.\-]")


def _make_spdxid(pkg: Package, seen: set[str]) -> str:
    """Produce a unique, spec-compliant SPDXID for a package."""
    safe_name = _SPDXID_RE.sub("-", pkg.name)
    safe_type = _SPDXID_RE.sub("-", pkg.pkg_type)
    base_id = f"SPDXRef-{safe_name}-{safe_type}"
    candidate = base_id
    counter = 1
    while candidate in seen:
        candidate = f"{base_id}-{counter}"
        counter += 1
    seen.add(candidate)
    return candidate


def _pkg_to_spdx(
    pkg: Package,
    change_type: str,
    now: str,
    seen_ids: set[str],
    from_version: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return (spdx_package_dict, spdxid)."""
    spdxid = _make_spdxid(pkg, seen_ids)
    license_declared = " AND ".join(pkg.licenses) if pkg.licenses else "NOASSERTION"

    entry: dict[str, Any] = {
        "SPDXID": spdxid,
        "name": pkg.name,
        "versionInfo": pkg.version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": license_declared,
        "copyrightText": "NOASSERTION",
    }

    if pkg.purl:
        entry["externalRefs"] = [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": pkg.purl,
            }
        ]

    delta_info: dict[str, str] = {"change": change_type}
    if from_version is not None:
        delta_info["previous-version"] = from_version

    annotations = [
        {
            "annotationType": "OTHER",
            "annotator": "Tool: arduino-sbom-delta",
            "annotationDate": now,
            "comment": json.dumps(delta_info),
        },
    ]
    entry["annotations"] = annotations

    return entry, spdxid


def print_delta_summary(
    added: PkgMap,
    updated: UpdateMap,
    removed: PkgMap,
) -> None:
    """Print a human-readable delta summary to stderr."""
    print(f"\n{'=' * 72}", file=sys.stderr)
    print(
        f"SBOM Delta — {len(added)} added, {len(updated)} updated, {len(removed)} removed",
        file=sys.stderr,
    )
    print(f"{'=' * 72}", file=sys.stderr)

    if added:
        print(f"\n[+] ADDED ({len(added)})", file=sys.stderr)
        for (_, pkg_type), pkg in sorted(added.items()):
            print(f"    {pkg.name:<42} {pkg.version:<22} [{pkg_type}]", file=sys.stderr)

    if updated:
        print(f"\n[~] UPDATED ({len(updated)})", file=sys.stderr)
        for (_, pkg_type), (old, new) in sorted(updated.items()):
            print(f"    {old.name:<42} {old.version} → {new.version} [{pkg_type}]", file=sys.stderr)

    if removed:
        print(f"\n[-] REMOVED ({len(removed)})", file=sys.stderr)
        for (_, pkg_type), pkg in sorted(removed.items()):
            print(f"    {pkg.name:<42} {pkg.version:<22} [{pkg_type}]", file=sys.stderr)

    print(f"\nTotal: {len(added) + len(updated) + len(removed)} changed packages\n", file=sys.stderr)


def build_delta_report(
    base_sbom_path: Path,
    full_sbom_path: Path,
    name: str,
    version: str,
    base_image: str | None = None,
    target_image: str | None = None,
) -> dict[str, Any]:
    """Compute the SBOM delta between two Syft native JSON documents.

    Returns a valid SPDX 2.3 JSON document containing only the packages
    added or updated in the target image relative to the base.
    Each delta package includes annotations for change-type, package-type,
    and (for updates) the previous-version.
    Removed packages are reported to stderr.
    """
    added, updated, removed = compute_delta(base_sbom_path, full_sbom_path)
    print_delta_summary(added, updated, removed)

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    doc_namespace = f"https://arduino.cc/sbom/delta/{name}-{version}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    container_spdxid = "SPDXRef-Container"
    seen_ids: set[str] = {"SPDXRef-DOCUMENT", container_spdxid}

    packages: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    # Container package (the thing the delta describes)
    packages.append({
        "SPDXID": container_spdxid,
        "name": name,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "comment": f"Delta SBOM for {target_image} — changes relative to base image {base_image}",
    })
    relationships.append({
        "spdxElementId": "SPDXRef-DOCUMENT",
        "relationshipType": "DESCRIBES",
        "relatedSpdxElement": container_spdxid,
    })

    # Added packages
    for key, pkg in sorted(added.items()):
        entry, spdxid = _pkg_to_spdx(pkg, "added", now, seen_ids)
        packages.append(entry)
        relationships.append({
            "spdxElementId": container_spdxid,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": spdxid,
        })

    # Updated packages
    for key, (old_pkg, new_pkg) in sorted(updated.items()):
        entry, spdxid = _pkg_to_spdx(new_pkg, "updated", now, seen_ids, from_version=old_pkg.version)
        packages.append(entry)
        relationships.append({
            "spdxElementId": container_spdxid,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": spdxid,
        })

    # Removed packages
    for key, pkg in sorted(removed.items()):
        entry, spdxid = _pkg_to_spdx(pkg, "removed", now, seen_ids)
        packages.append(entry)
        relationships.append({
            "spdxElementId": container_spdxid,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": spdxid,
        })

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": name,
        "documentNamespace": doc_namespace,
        "creationInfo": {
            "created": now,
            "creators": [
                "Tool: arduino-sbom-delta",
                "Organization: Arduino SRL",
            ],
        },
        "documentDescribes": [container_spdxid],
        "packages": packages,
        "relationships": relationships,
    }


def write_output(output_path: Path, content: str) -> None:
    """Write content to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def require_command(command: str, install_hint: str) -> None:
    """Ensure an external command exists on PATH."""
    if shutil.which(command) is None:
        raise SbomDeltaError(f"'{command}' not found. {install_hint}")


def parse_container_spec(spec: str, default_version: str) -> tuple[str, str]:
    """Split a ``name[:version]`` CLI token, falling back to the default version."""
    name, _, version = spec.partition(":")
    if not name:
        raise SbomDeltaError(f"Invalid container spec '{spec}': expected name[:version].")
    return name, version or default_version


def build_container_image(registry: str, container: str, version: str) -> str:
    """Build the fully qualified image reference for a container."""
    return f"{normalize_registry(registry)}app-bricks/{container}:{version}"


PLATFORM = "linux/arm64"


def extract_attested_sbom(image: str, payload: object) -> dict[str, Any]:
    """Return the SPDX document from the ``imagetools inspect --format '{{ json .SBOM }}'`` output.

    A single-platform image exposes the document directly, a multi-platform one
    keys it by platform; the images built here have exactly one.
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("SPDX"), dict):
            return payload["SPDX"]
        documents = [entry["SPDX"] for entry in payload.values() if isinstance(entry, dict) and isinstance(entry.get("SPDX"), dict)]
        if len(documents) == 1:
            return documents[0]
        if documents:
            raise SbomDeltaError(f"image '{image}' carries {len(documents)} SBOM attestations, expected one platform")
    raise SbomDeltaError(f"image '{image}' carries no SBOM attestation")


def fetch_attested_sbom(image: str, output_path: Path) -> None:
    """Save the SPDX SBOM attested to a published image, generated by BuildKit at build time."""
    print(f"    attestation: {image}", file=sys.stderr)
    require_command("docker", "Install Docker with buildx.")
    command = ["docker", "buildx", "imagetools", "inspect", image, "--format", "{{ json .SBOM }}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or f"exit status {result.returncode}"
        raise SbomDeltaError(f"failed to inspect image '{image}': {details}")
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        raise SbomDeltaError(f"unreadable SBOM attestation of '{image}': {exc}") from exc
    document = extract_attested_sbom(image, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scan_image(image: str, output_path: Path) -> None:
    """Scan an image with Syft and save the SPDX JSON report."""
    print(f"    syft scan: {image}", file=sys.stderr)
    require_command("syft", "Install from: https://github.com/anchore/syft#installation")

    commands = [
        ["syft", f"registry:{image}", "--platform", PLATFORM, "--output", f"spdx-json={output_path}", "--quiet"],
        ["syft", image, "--platform", PLATFORM, "--output", f"spdx-json={output_path}", "--quiet"],
    ]

    last_result: subprocess.CompletedProcess[str] | None = None
    for cmd in commands:
        last_result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if last_result.returncode == 0:
            # Re-write as pretty-printed JSON
            raw = load_json(output_path)
            output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return

    details = (last_result.stderr or last_result.stdout or "").strip() if last_result else ""
    if not details and last_result:
        details = f"syft exited with status {last_result.returncode}"
    raise SbomDeltaError(f"failed to scan image '{image}': {details}")


def generate_delta_for_container(
    containers: Containers,
    output_dir: Path,
    container: str,
    registry: str,
    version: str,
) -> None:
    """Generate delta SBOM artifacts for a single container."""
    base_image = resolve_runtime_base(containers=containers, container=container, registry=registry, version=version)
    target_image = build_container_image(registry=registry, container=container, version=version)

    print(f"[{container}]")
    print(f"  base      : {base_image}")
    print(f"  container : {target_image}")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_sbom = output_dir / "base.spdx.json"
    full_sbom = output_dir / "full.spdx.json"

    if containers.parent[container]:
        print("  reading the base attestation...")
        fetch_attested_sbom(image=base_image, output_path=base_sbom)
    else:
        print("  scanning the external base...")
        scan_image(image=base_image, output_path=base_sbom)
    print("  reading the container attestation...")
    fetch_attested_sbom(image=target_image, output_path=full_sbom)

    print("  computing delta...")
    report = build_delta_report(
        base_sbom_path=base_sbom,
        full_sbom_path=full_sbom,
        name=container,
        version=version,
        base_image=base_image,
        target_image=target_image,
    )

    output_path = output_dir / "delta.spdx.json"
    write_output(output_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"  delta -> {output_path}")


def run_generate(args: argparse.Namespace) -> int:
    """Run the end-to-end delta generation workflow."""
    try:
        containers = Containers(REPO_ROOT / "containers")
    except ContainerDepsError as exc:
        raise SbomDeltaError(str(exc)) from exc
    if args.containers:
        specs = [parse_container_spec(spec, args.version) for spec in args.containers]
    else:
        specs = [(name, args.version) for name in containers.names]

    registry = normalize_registry(args.registry)
    output_root = Path(args.output_dir)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" SBOM Delta Generator  |  {registry}  ({PLATFORM})  ->  {output_root}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    failed: list[str] = []
    for container, version in specs:
        try:
            generate_delta_for_container(
                containers=containers,
                output_dir=output_root / f"{container}-{version}",
                container=container,
                registry=registry,
                version=version,
            )
        except SbomDeltaError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append(f"{container}:{version}")
        print("")

    if failed:
        raise SbomDeltaError(f"Failed: {' '.join(failed)}")

    print("Done.")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("containers", nargs="*", help="Containers as name[:version] (default: every container, at --version).")
    parser.add_argument("--registry", default=os.environ.get("REGISTRY", "ghcr.io/arduino/"), help="Registry prefix.")
    parser.add_argument("--version", default=os.environ.get("VERSION", "latest"), help="Image tag to scan for containers given without a version.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "sbom-delta"),
        help="Directory receiving one <name>-<version>/ folder per scanned container (default: ./sbom-delta).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = create_parser().parse_args(sys.argv[1:] if argv is None else argv)

    try:
        return run_generate(args)
    except SbomDeltaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
