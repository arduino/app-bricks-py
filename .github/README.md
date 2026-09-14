# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/<group>/<name>/`. Each container is described by a `ci.json` file in its directory that drives CI behaviour — no workflow changes are needed when adding a new container.

### Layout

Containers are filed under three groups, which document what they are for:

| Group | Contains |
|---|---|
| `containers/ai/` | AI/ML model runners |
| `containers/bricks/` | Images shipping the library itself and its supporting tooling |
| `containers/base/` | Shared base images other containers derive `FROM` (`base_image: true`) |

The group is **not** part of a container's identity: a container is always referred to by its leaf
directory name, which is also its image name (`ghcr.io/arduino/app-bricks/<name>`) and the value used in
`downstream`, in the build matrices and in the `containers` input of the dev workflow. CI locates a
container by globbing `containers/*/<name>/ci.json`, so moving a container between groups only means
updating its `watch_paths`. Leaf names must stay unique across groups; the build planner fails loudly if
two groups declare the same name.

The full list of images, with what each one builds from and what it is for, is the inventory in
[containers/README.md](../containers/README.md#inventory).

## Release Workflow

A single workflow (`docker-publish.yml`) is triggered by any `release/X.Y.Z` tag, or manually via
`workflow_dispatch`. **Every release publishes every container at `X.Y.Z`**, together with the Python
`.whl` and `sboms.zip` on the GitHub Release. There is one release cycle: the library and the containers
it runs always ship together.

Containers flagged `base_image` are not release targets by themselves: they are rebuilt, and tagged with
the release version, as the base of the images that derive from them.

### Tagging

Every released image receives a `:<version>` tag. A `:latest` tag is also applied unless the version
contains `rc`, `alpha` or `beta`, which marks a prerelease.

### Compose file versioning

Bricks and services ship compose files describing the containers they need. Their image references carry
the `__BRICKS_RELEASE_VERSION__` placeholder instead of a version, and so do the `models/models-*.yaml`
files. `arduino-bricks-release` stamps the release version into the copies bundled in the wheel at build
time, and `arduino-bricks-list-modules` does the same when it provisions the compose files at runtime. The
version comes from `BRICKS_RELEASE_VERSION`, which `python-apps-base` sets to the tag it was built with,
or from the installed library otherwise. Compose files therefore always reference the containers published
by the same release. A version can still be written literally where a specific image is really needed.

## Dependency Ordering

The set of containers to build is every non-base container plus the transitive closure of its bases, so
everything being rebuilt sits on a freshly built base. The result is split into topological waves by
`scripts/build_levels.py` — `level_0` has no in-set parent, each later wave depends on the previous one —
and `docker-publish.yml` chains one `build-l<n>` job per wave with `needs`.

So a release builds `python-slim` → (`python-base`, `models-downloader`, `llamacpp-runner`) →
`python-apps-base`, and `qairt-common-base` → (`aihub-models-runner`, `llamacpp-npu-runner`) →
(`gesture-recognition-runner`, `pose-estimation-runner`), each chain in that order. Dependencies are
declared by the `downstream` field in `ci.json`, which must mirror the `FROM` lines of the Dockerfiles.

Only the `level_0` wave may skip its build (see [Skip-Rebuild Logic](#skip-rebuild-logic)); later waves
always rebuild, since their base image was just rebuilt.

## Adding a New Container

1. Create `containers/<group>/my-container/Dockerfile`, filing it under the group that describes what it
   is for — see [Layout](#layout).
2. Create `containers/<group>/my-container/ci.json`:

```json
{
  "watch_paths": ["containers/<group>/my-container/"],
  "build_whl": false,
  "build_args": {},
  "sbom": { "runtime_base": "${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}" },
  "downstream": []
}
```

3. If the image installs Python packages, declare them in a `pyproject.toml` locked by `task deps:lock`
   and register the container in the dependency license scan, see
   [scripts/licensed/README.md](../scripts/licensed/README.md).
4. Push a `release/X.Y.Z` tag — the workflow picks up every container automatically.

To declare that another container depends on yours, add it to `downstream`:

```json
"downstream": ["my-other-container"]
```

> **Note**: any container listed in `downstream` must declare `ARG REGISTRY` and `ARG BASE_IMAGE_VERSION` in its Dockerfile. The CI passes the upstream image's tag via `BASE_IMAGE_VERSION` so the downstream image pulls the freshly built version, not `latest`.

No workflow file changes required.

## ci.json Reference

| Field | Type | Description |
|---|---|---|
| `watch_paths` | string[] | Repo-relative paths checked by the skip-rebuild logic — must include the container's own directory |
| `base_image` | bool | Shared base image: never a release target by itself, only rebuilt as a dependency |
| `build_whl` | bool | Download the `.whl` artifact into the build context before the Docker build (set `true` on containers that install it) |
| `build_args` | object | Docker build args passed to the Dockerfile (key/value pairs) |
| `sbom.runtime_base` | string | Image the delta SBOM is computed against — must match the Dockerfile's `FROM` |
| `downstream` | string[] | Containers that depend on this one — rebuilt automatically after this container is built |

## SBOMs

`sboms.zip`, attached to every GitHub Release, is generated from the published images, so nothing SBOM-related lives in the tree. Every image is built by the release, so the archive covers the whole build set at the release version. `scripts/sbom_delta.py` scans one `name:version` with Syft against its `sbom.runtime_base` and writes `<name>-<version>/{base,full,delta}.spdx.json`.

Scanning is spread over `_sbom-wave.yml` jobs, one matrix leg per image: `sbom-l<n>` starts as soon as `build-l<n>` is pushed and overlaps with the next wave's build. Each leg uploads a `sbom-delta-<name>-<version>` artifact; a failed scan is a warning, never a failure. `upload-release` collects the artifacts, checks them against the build set, writes any gap to `MISSING.txt` inside the archive and to the job summary, and attaches the zip.

The dev workflow runs the same scan per built image through `.github/actions/sbom-delta` and uploads it as a `sbom-delta-<name>-<tag>` run artifact.

## Skip-Rebuild Logic

For `level_0` containers only, the release checks whether the container's `watch_paths` actually changed since the previous `release/*` tag:

- **Changed** → full Docker build and push
- **Unchanged** → `crane copy` re-tags the existing image to the new version (instant, no rebuild), and to `latest` unless the version is a prerelease

So releasing `release/X.Y.Z` when nothing under `containers/base/python-slim/` changed re-tags `python-slim` from the previous release instead of rebuilding it.

## Dev Build Workflow

`docker-build.yml` ("DEV - Build & Publish Branch Containers") is triggered manually via `workflow_dispatch` with:

- `branch` — branch to build (defaults to the branch the workflow is run from)
- `containers` — comma-separated list of containers to build, or `all` (default)
- `tag` — optional custom image tag
- `skip_cache` — rebuild without cache

Images are tagged `dev-<branch-name>` (branch name lowercased and sanitized, e.g. `feat/My-Feature` → `dev-feat-my-feature`), plus a run-number-suffixed alias (e.g. `dev-feat-my-feature-42`), unless a custom `tag` is provided.

**Dependency ordering**: the same topological planner as the release (`scripts/build_levels.py`, in `--mode dev`) expands the selection with its ancestors and descendants and splits it into waves — `build-l0`, `build-l1`, `build-l2` — where each wave waits for the previous one and receives `BASE_IMAGE_VERSION=<image-tag>` as a build arg so it uses the freshly built upstream images. The ordering is driven entirely by the `downstream` field in ci.json — no hardcoded container names in the workflow.

**Wheel**: containers with `build_whl` get a wheel built with `BRICKS_RELEASE_VERSION=<image-tag>`, so the compose files it bundles reference the dev images of the same run.

## Image Cleanup

`docker-cleanup.yml` runs two independent jobs:

| Trigger | Job | What it does |
|---|---|---|
| Branch deletion | `cleanup` | Deletes every GHCR version tagged `dev-<deleted-branch>`, including the run-number aliases and the build cache |
| Weekly (Sunday 03:00 UTC) or manual | `prune-untagged` | Deletes untagged container versions, the blobs orphaned by overwritten buildx cache manifests. Any tagged version is preserved, and each candidate is re-checked right before deletion. The manual run accepts a `dry_run` flag to only list what would be deleted. |

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: Buildx registry cache (`<image>:buildcache`, `mode=max`), invalidated by content, `skip_cache` forces a cold rebuild
- **Release assets**: The `.whl` and `sboms.zip` are uploaded to the GitHub Release via `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`, measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR is found, it falls back to the GitHub Actions Job Summary.
