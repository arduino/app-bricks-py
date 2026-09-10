#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Builds one venv per app declared in .licensed.yml, then runs licensed cache and status on each."""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import yaml

SRC = Path("/src")
CONFIG = SRC / ".licensed.yml"
VENVS = Path("/venvs")
PIP_CACHE = VENVS / ".pip-cache"
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
        if "venv" not in app:
            fail(f"app {app['name']} has no venv section in .licensed.yml, nothing to scan")
        if ("requirements" in app["venv"]) == ("project" in app["venv"]):
            fail(f"app {app['name']} needs exactly one of venv.requirements or venv.project")
    return config, apps


def requirement_lines(path):
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


def check_requirements_covered(apps):
    """Every non-empty requirements file under containers/ must belong to a scanned app."""
    declared = {str(SRC / app["venv"]["requirements"]) for app in apps if "requirements" in app["venv"]}
    found = glob.glob(str(SRC / "containers/*/*/requirements*.txt"))
    missing = sorted(Path(f).relative_to(SRC) for f in found if f not in declared and requirement_lines(Path(f)))
    if missing:
        fail("requirements files not covered by any app in .licensed.yml:\n  " + "\n  ".join(map(str, missing)))


def venv_inputs(app):
    """Files whose content decides what the venv contains."""
    venv = app["venv"]
    if "requirements" in venv:
        return [SRC / venv["requirements"]]
    return [SRC / venv["project"] / "pyproject.toml"]


def venv_key(app):
    digest = hashlib.sha256(sys.version.encode())
    for path in venv_inputs(app):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def install(app, pip):
    venv = app["venv"]
    if "requirements" in venv:
        subprocess.run([*pip, "-r", str(SRC / venv["requirements"])], check=True)
        return
    # Install the project from a copy of its build files only, so the sources stay untouched
    project = SRC / venv["project"]
    pyproject = tomllib.loads((project / "pyproject.toml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(project / "pyproject.toml", tmp)
        for backend_path in pyproject.get("build-system", {}).get("backend-path", []):
            shutil.copytree(project / backend_path, Path(tmp) / backend_path, ignore=shutil.ignore_patterns("__pycache__"))
        extras = ",".join(venv.get("extras", []))
        spec = f"{tmp}[{extras}]" if extras else tmp
        env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"}
        subprocess.run([*pip, spec], check=True, env=env)


def build_venv(app):
    venv_dir = Path(app["python"]["virtual_env_dir"])
    key_file = venv_dir / KEY_FILE
    key = venv_key(app)
    if key_file.exists() and key_file.read_text() == key:
        return f"{app['name']}: venv reused"
    shutil.rmtree(venv_dir, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    pip = [str(venv_dir / "bin/pip"), "install", "-q", "--cache-dir", str(PIP_CACHE)]
    install(app, pip)
    for entry in venv_dir.glob("lib/python*/site-packages/*"):
        if entry.name not in KEEP_IN_SITE_PACKAGES and not entry.name.endswith(".dist-info"):
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    key_file.write_text(key)
    return f"{app['name']}: venv built"


def run_licensed(config, app, tmp):
    """Runs licensed on a single app through a config holding only that app."""
    app_config = {**config, "root": str(SRC), "apps": [app]}
    config_file = Path(tmp) / f"{app['name']}.yml"
    config_file.write_text(yaml.safe_dump(app_config))
    output = []
    ok = True
    for command in ("cache", "status"):
        result = subprocess.run(["licensed", command, "-c", str(config_file)], cwd=SRC, capture_output=True, text=True)
        output.append(result.stdout + result.stderr)
        if result.returncode != 0:
            ok = False
            break
    with print_lock:
        print(f"==> {app['name']}\n" + "".join(output), flush=True)
    return ok


def main():
    config, apps = load_config()
    check_requirements_covered(apps)
    VENVS.mkdir(exist_ok=True)
    with ThreadPoolExecutor() as pool:
        for message in pool.map(build_venv, apps):
            print(message, flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            results = list(pool.map(lambda app: run_licensed(config, app, tmp), apps))
    if not all(results):
        failed = [app["name"] for app, ok in zip(apps, results) if not ok]
        fail("licensed reported problems for: " + ", ".join(failed))


if __name__ == "__main__":
    main()
