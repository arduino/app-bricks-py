#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

"""Compute delta reports between container SBOMs"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def load_container_config(containers_dir: Path, container: str) -> dict[str, Any]:
    """Load ``ci.json`` metadata for a container."""
    return load_json(containers_dir / container / "ci.json")


def build_resolution_context(
    config: dict[str, Any],
    registry: str,
    version: str,
    build_args: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the variable map used to resolve runtime base templates."""
    default_build_args = config.get("build_args") or {}
    if not isinstance(default_build_args, dict):
        raise SbomDeltaError("Expected 'build_args' to be a JSON object.")

    context = {str(k): str(v) for k, v in default_build_args.items()}
    context["REGISTRY"] = normalize_registry(registry)
    context["VERSION"] = version
    context["BASE_IMAGE_VERSION"] = version

    for key, value in (build_args or {}).items():
        context[str(key)] = str(value)

    context["REGISTRY"] = normalize_registry(context["REGISTRY"])
    return context


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


def resolve_runtime_base(
    containers_dir: Path,
    container: str,
    registry: str,
    version: str,
    build_args: dict[str, str] | None = None,
) -> str:
    """Resolve the fully qualified runtime base image for a container.

    Uses the ``sbom.runtime_base`` template in the container's ``ci.json``,
    expanding ``${VAR}`` placeholders with registry, version, and build args.
    """
    config = load_container_config(containers_dir, container)
    sbom_config = config.get("sbom")
    if not isinstance(sbom_config, dict):
        raise SbomDeltaError(f"Container '{container}' is missing 'sbom.runtime_base'.")

    runtime_base = sbom_config.get("runtime_base")
    if not isinstance(runtime_base, str) or not runtime_base.strip():
        raise SbomDeltaError(f"Container '{container}' is missing 'sbom.runtime_base'.")

    context = build_resolution_context(config=config, registry=registry, version=version, build_args=build_args)
    return resolve_template(runtime_base.strip(), context)


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

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    doc_namespace = f"https://arduino.cc/sbom/delta/{name}-{version}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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


def discover_containers(containers_dir: Path) -> list[str]:
    """Discover containers by looking for ``ci.json`` files."""
    return sorted(p.parent.name for p in containers_dir.glob("*/ci.json") if p.is_file())


def build_container_image(registry: str, container: str, version: str) -> str:
    """Build the fully qualified image reference for a container."""
    return f"{normalize_registry(registry)}app-bricks/{container}:{version}"


PLATFORM = "linux/arm64"


def scan_image(image: str, output_path: Path) -> None:
    """Scan an image with Syft and save the SPDX JSON report."""
    print(f"    syft scan: {image}", file=sys.stderr)

    commands = [
        ["syft", f"registry:{image}", "--platform", PLATFORM, "-o", "spdx-json", "--file", str(output_path), "--quiet"],
        ["syft", image, "--platform", PLATFORM, "-o", "spdx-json", "--file", str(output_path), "--quiet"],
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
    containers_dir: Path,
    output_dir: Path,
    container: str,
    registry: str,
    version: str,
) -> None:
    """Generate delta SBOM artifacts for a single container."""
    base_image = resolve_runtime_base(containers_dir=containers_dir, container=container, registry=registry, version=version)
    target_image = build_container_image(registry=registry, container=container, version=version)

    print(f"[{container}]")
    print(f"  base      : {base_image}")
    print(f"  container : {target_image}")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_sbom = output_dir / "base.spdx.json"
    full_sbom = output_dir / "full.spdx.json"

    print("  scanning base...")
    scan_image(image=base_image, output_path=base_sbom)
    print("  scanning container...")
    scan_image(image=target_image, output_path=full_sbom)

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
    require_command("syft", "Install from: https://github.com/anchore/syft#installation")

    containers_dir = REPO_ROOT / "containers"
    containers = list(args.containers) if args.containers else discover_containers(containers_dir)
    if not containers:
        raise SbomDeltaError(f"No containers found (looked for ci.json under {containers_dir}).")

    registry = normalize_registry(args.registry)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" SBOM Delta Generator  |  {registry}  v{args.version}  ({PLATFORM})")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    failed: list[str] = []
    for container in containers:
        output_dir = containers_dir / container / "sbom-delta"
        try:
            generate_delta_for_container(
                containers_dir=containers_dir,
                output_dir=output_dir,
                container=container,
                registry=registry,
                version=args.version,
            )
        except SbomDeltaError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append(container)
        print("")

    if failed:
        raise SbomDeltaError(f"Failed: {' '.join(failed)}")

    print("Done.")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("containers", nargs="*", help="Container names (default: all with ci.json).")
    parser.add_argument("--registry", default=os.environ.get("REGISTRY", "ghcr.io/arduino/"), help="Registry prefix.")
    parser.add_argument("--version", default=os.environ.get("VERSION", "latest"), help="Image tag to scan.")
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
