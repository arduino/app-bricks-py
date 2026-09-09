# Dependency license scan

`task license:deps` checks the license of every Python package we ship, running [licensed](https://github.com/licensee/licensed) inside Docker. Nothing but Docker is needed on the host.

## What it does

1. Builds the `arduino-licensed` image from the [Dockerfile](Dockerfile). One stage per scanned app creates a venv from that app's requirements file and keeps only the package metadata licensed reads.
2. Runs `licensed cache`, which writes one record per package under `.licenses/<app>/pip/`.
3. Runs `licensed status`, which fails when a record is missing, stale, unreviewed or carries a license outside the allowed list.

The apps, the allowed licenses and the manually reviewed packages are declared in [.licensed.yml](../../.licensed.yml). Each app maps to `/venvs/<app>` inside the image.

## What is scanned

Every container that installs Python packages, under its own name. `python-apps-base` stands for the library with all its extras, which is exactly what that image installs. Containers without Python packages, such as the Edge Impulse and Qualcomm images, have nothing to scan. System packages are covered by the SBOMs generated at release, not by this scan.

## Adding a container

1. List its Python packages in a `requirements.txt` in the container directory and install from that file in its Dockerfile. Inline `pip install <package>` lines are invisible to the scan.
2. Add an app entry in `.licensed.yml` with `virtual_env_dir: "/venvs/<name>"`.
3. Add a venv stage in the [Dockerfile](Dockerfile) that copies the requirements file, its `COPY --from` line in the final stage, and the file in [Dockerfile.dockerignore](Dockerfile.dockerignore).
4. Run `task license:deps` and commit the new records under `.licenses/<name>/`.

## When the check fails

- **cached dependency record out of date**: a package version changed. `licensed cache` has already rewritten the record, review the diff and commit it.
- **license needs review**: licensee could not classify the license text. Read the text in the record and, if the license is acceptable, add the package to `reviewed` in `.licensed.yml` with a comment naming the license.
- **missing license text**: the wheel ships no license file. Fill the `licenses` block of the record by hand from the project's license and say where it came from in `sources`. Licensed keeps hand edits until the version changes.
- **license text has changed**: read the new text, then remove `review_changed_license: true` from the record.

## Caches

Docker caches the layers and the pip downloads, so a run with no changes takes about 20 seconds and a cold build about 2 minutes. `task license:deps:clean` removes only this image's build cache and keeps the image. It relies on a naming convention: every stage is named `licensed-...` and every RUN goes through one of the `licensed-*` helpers in [bin/](bin/), so the word appears in every cache record. An inline RUN would escape the filter.

## Versions

`licensed` and `licensee` are pinned in the Dockerfile. Licensee is what classifies license texts, so bumping it can change existing verdicts. Re-run the scan after a bump and review the diff.
