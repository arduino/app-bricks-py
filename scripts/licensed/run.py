#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Installs the locked dependencies of every app in .licensed.yml into its venv, then runs licensed cache and status on each."""

import glob
import hashlib
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import yaml

SRC = Path("/src")
CONFIG = SRC / ".licensed.yml"
VENVS = Path("/venvs")
KEY_FILE = ".key"
# Package metadata licensed reads, everything else is pruned from the venvs
KEEP_IN_SITE_PACKAGES = {"pip", "setuptools", "wheel", "pkg_resources", "_distutils_hack"}

print_lock = Lock()


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def load_config():
    config = yaml.safe_load(CONFIG.read_text())
    apps = config.get("apps") or []
    for app in apps:
        if "project" not in app.get("venv", {}):
            fail(f"app {app['name']} has no venv.project in .licensed.yml, nothing to scan")
    return config, apps


def project_dir(app):
    return SRC / app["venv"]["project"]


def requirement_lines(path):
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def check_projects_covered(apps):
    """Every uv project under containers/ must belong to a scanned app, and no requirements file may install packages."""
    declared = {project_dir(app).resolve() for app in apps}
    found = [Path(f).parent for f in glob.glob(str(SRC / "containers/*/*/pyproject.toml"))]
    missing = sorted(p.relative_to(SRC) for p in found if p.resolve() not in declared)
    if missing:
        fail("uv projects not covered by any app in .licensed.yml:\n  " + "\n  ".join(map(str, missing)))
    requirements = glob.glob(str(SRC / "containers/*/*/requirements*.txt"))
    listing = sorted(Path(f).relative_to(SRC) for f in requirements if requirement_lines(Path(f)))
    if listing:
        fail("requirements files are not scanned, declare the packages in the container's pyproject.toml:\n  " + "\n  ".join(map(str, listing)))


def check_lock(app):
    """A lock that lags its pyproject would make the scan see something else than the image installs."""
    result = subprocess.run(["uv", "lock", "--check", "--project", str(project_dir(app))], capture_output=True, text=True)
    return None if result.returncode == 0 else f"{app['name']}: {result.stderr.strip()}"


def venv_key(app):
    digest = hashlib.sha256(sys.version.encode())
    digest.update((project_dir(app) / "uv.lock").read_bytes())
    return digest.hexdigest()


def build_venv(app):
    """Installs exactly what the image installs, the same export and hash checked install the Dockerfiles run."""
    venv_dir = Path(app["python"]["virtual_env_dir"])
    key_file = venv_dir / KEY_FILE
    key = venv_key(app)
    if key_file.exists() and key_file.read_text() == key:
        return f"{app['name']}: venv reused"
    shutil.rmtree(venv_dir, ignore_errors=True)
    # The seed gives the venv the pip licensed reads package metadata with
    subprocess.run(["uv", "venv", "-q", "--seed", str(venv_dir)], check=True)
    export = ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(project_dir(app))]
    export += [f"--extra={extra}" for extra in app["venv"].get("extras", [])]
    requirements = subprocess.run(export, check=True, capture_output=True, text=True).stdout
    install = ["uv", "pip", "install", "-q", "--python", str(venv_dir / "bin/python"), "--require-hashes", "-r", "-"]
    subprocess.run(install, input=requirements, check=True, text=True)
    for entry in venv_dir.glob("lib/python*/site-packages/*"):
        if entry.name not in KEEP_IN_SITE_PACKAGES and not entry.name.endswith(".dist-info"):
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    key_file.write_text(key)
    return f"{app['name']}: venv built"


def run_licensed(config, app, tmp, commands=("cache", "status")):
    """Runs licensed on a single app through a config holding only that app."""
    app_config = {**config, "root": str(SRC), "apps": [app]}
    config_file = Path(tmp) / f"{app['name']}.yml"
    config_file.write_text(yaml.safe_dump(app_config))
    output = []
    ok = True
    for command in commands:
        result = subprocess.run(["licensed", command, "-c", str(config_file)], cwd=SRC, capture_output=True, text=True)
        output.append(result.stdout + result.stderr)
        if result.returncode != 0:
            ok = False
            break
    with print_lock:
        print(f"==> {app['name']}\n" + "".join(output), flush=True)
    return ok


def main():
    # --check reports the status of the committed records without rewriting them
    commands = ("status",) if "--check" in sys.argv[1:] else ("cache", "status")
    config, apps = load_config()
    check_projects_covered(apps)
    VENVS.mkdir(exist_ok=True)
    with ThreadPoolExecutor() as pool:
        stale = [problem for problem in pool.map(check_lock, apps) if problem]
        if stale:
            fail("uv.lock is out of date, run task deps:lock:\n  " + "\n  ".join(stale))
        for message in pool.map(build_venv, apps):
            print(message, flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            results = list(pool.map(lambda app: run_licensed(config, app, tmp, commands), apps))
    if not all(results):
        failed = [app["name"] for app, ok in zip(apps, results) if not ok]
        fail("licensed reported problems for: " + ", ".join(failed))


if __name__ == "__main__":
    main()
