# Dependency license scan

`task fix:licenses` records the license of every Python package we ship and `task check:licenses` verifies the records without touching them, both running [licensed](https://github.com/licensee/licensed) inside Docker. Nothing but Docker is needed on the host.

## What it does

1. Builds the `arduino-licensed` image from the [Dockerfile](Dockerfile). It holds licensed, the compilers some packages need and the same Python as the containers. It knows nothing about the apps.
2. Runs [run.py](run.py) with the repository mounted at `/src` and the `arduino-licensed-venvs` volume at `/venvs`. For each app in [.licensed.yml](../../.licensed.yml) it creates a venv from the uv project named in `venv.project`, running the same `uv export` and hash checked `uv pip install` the Dockerfiles run, and keeps only the package metadata licensed reads. A venv is rebuilt only when the lock or the Python version change.
3. Runs `licensed cache` and `licensed status` on every app, all apps in parallel (`status` only with `--check`, what `task check:licenses` passes). `cache` writes one record per package under `.licenses/<app>/pip/`, `status` fails when a record is missing, stale, unreviewed or carries a license outside the allowed list.

Before scanning, run.py refuses to start if an app has no `venv.project`, if a `pyproject.toml` under `containers/` belongs to no app, if a `requirements*.txt` under `containers/` lists packages, or if a `uv.lock` lags its `pyproject.toml`. A container cannot slip out of the scan, and the scan cannot see anything else than what the images install.

## What is scanned

Every container that installs Python packages, under its own name. `python-apps-base` stands for the library with all its extras, which is exactly what that image installs. Containers without Python packages, such as the Edge Impulse and Qualcomm images, have nothing to scan. System packages are covered by the SBOMs generated at release, not by this scan.

## Adding a container

1. Declare its Python packages in a `pyproject.toml` in the container directory, run `task deps:lock`, and install from `uv.lock` in its Dockerfile the way the other containers do. Inline `pip install <package>` lines are invisible to the scan.
2. Add an app entry in `.licensed.yml` with `virtual_env_dir: "/venvs/<name>"` and `venv.project` pointing at the directory.
3. Run `task fix:licenses` and commit the new records under `.licenses/<name>/`.

## When the check fails

- **cached dependency record out of date**: a lock moved a package to another version. `licensed cache` has already rewritten the record, review the diff and commit it.
- **license needs review**: licensee could not classify the license text. Read the text in the record and, if the license is acceptable, add the package to `reviewed` in `.licensed.yml` with a comment naming the license.
- **missing license text**: the wheel ships no license file. Fill the `licenses` block of the record by hand from the project's license and say where it came from in `sources`. Licensed keeps hand edits until the version changes.
- **license text has changed**: read the new text, then remove `review_changed_license: true` from the record.

## Caches

The venvs and the uv cache live in the `arduino-licensed-venvs` Docker volume, nothing is written in the workspace. A run with no lock changes takes about 15 seconds, a cold one a few minutes. `task fix:licenses:clean` deletes the volume and keeps the image.

CI caches the image layers, through Buildx and the Actions cache, and the venvs, keyed by the content of every `uv.lock`. A venv is a pure function of its lock and the image's Python, and run.py rebuilds any venv whose key no longer matches, so a cached venv cannot pass a check against something the images do not install. The workflow also runs on pushes to main, whose caches every pull request can read.

## Versions

`licensed`, `licensee`, `uv` and the base image are pinned in the Dockerfile. Licensee is what classifies license texts, so bumping it can change existing verdicts. Re-run the scan after a bump and review the diff.
