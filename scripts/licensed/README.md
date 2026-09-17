# Dependency license scan

`task license:deps` checks the license of every Python package we ship, running [licensed](https://github.com/licensee/licensed) inside Docker. Nothing but Docker is needed on the host.

## What it does

1. Builds the `arduino-licensed` image from the [Dockerfile](Dockerfile). It holds licensed, the compilers some packages need and the same Python as the containers. It knows nothing about the apps.
2. Runs [run.py](run.py) with the repository mounted at `/src` and the `arduino-licensed-venvs` volume at `/venvs`. For each app in [.licensed.yml](../../.licensed.yml) it creates a venv from the files named in the app's `venv` section, keeping only the package metadata licensed reads. A venv is rebuilt only when those files or the Python version change.
3. Runs `licensed cache` and `licensed status` on every app, all apps in parallel. `cache` writes one record per package under `.licenses/<app>/pip/`, `status` fails when a record is missing, stale, unreviewed or carries a license outside the allowed list.

Before scanning, run.py refuses to start if an app has no `venv` section or if a non-empty `requirements*.txt` under `containers/` belongs to no app, so a container cannot slip out of the scan unnoticed.

## What is scanned

Every container that installs Python packages, under its own name. `python-apps-base` stands for the library with all its extras, which is exactly what that image installs. Containers without Python packages, such as the Edge Impulse and Qualcomm images, have nothing to scan. System packages are covered by the SBOMs generated at release, not by this scan.

## Adding a container

1. List its Python packages in a `requirements.txt` in the container directory and install from that file in its Dockerfile. Inline `pip install <package>` lines are invisible to the scan.
2. Add an app entry in `.licensed.yml` with `virtual_env_dir: "/venvs/<name>"` and `venv.requirements` pointing at the file.
3. Run `task license:deps` and commit the new records under `.licenses/<name>/`.

## When the check fails

- **cached dependency record out of date**: a package version changed. `licensed cache` has already rewritten the record, review the diff and commit it.
- **license needs review**: licensee could not classify the license text. Read the text in the record and, if the license is acceptable, add the package to `reviewed` in `.licensed.yml` with a comment naming the license.
- **missing license text**: the wheel ships no license file. Fill the `licenses` block of the record by hand from the project's license and say where it came from in `sources`. Licensed keeps hand edits until the version changes.
- **license text has changed**: read the new text, then remove `review_changed_license: true` from the record.

## Caches

The venvs and the pip download cache live in the `arduino-licensed-venvs` Docker volume, nothing is written in the workspace. A run with no changes takes about 15 seconds, a cold one a few minutes. `task license:deps:clean` deletes the volume and keeps the image.

CI caches only what cannot change the verdict: the image layers, through Buildx and the Actions cache, and the pip download cache. The venvs are rebuilt on every run, so each pull request resolves afresh. Requirements are not fully pinned, so a cached venv could otherwise pass a check against last month's resolution. The workflow also runs on pushes to main, whose caches every pull request can read.

## Versions

`licensed`, `licensee` and the base image are pinned in the Dockerfile. Licensee is what classifies license texts, so bumping it can change existing verdicts. Re-run the scan after a bump and review the diff.
