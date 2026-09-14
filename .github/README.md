# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/<group>/<name>/` and a matching target in `docker-bake.hcl` at the repository root. The workflows build through `docker buildx bake`, so adding a container never means changing a workflow.

### Layout

Containers are filed under three groups, which document what they are for:

| Group | Contains |
|---|---|
| `containers/ai/` | AI/ML model runners |
| `containers/bricks/` | Images shipping the library itself and its supporting tooling |
| `containers/base/` | Shared base images other containers derive `FROM` (`base_image: true`) |

The group is **not** part of a container's identity: a container is always referred to by its leaf
directory name, which is also its image name (`ghcr.io/arduino/app-bricks/<name>`), its bake target and
the value used in the `containers` input of the dev workflow. CI locates a container by globbing
`containers/*/<name>/Dockerfile`, so moving a container between groups only means updating the `context`
of its bake target. Leaf names must stay unique across groups; `scripts/container_deps.py` fails loudly
if two groups declare the same name.

The full list of images, with what each one builds from and what it is for, is the inventory in
[containers/README.md](../containers/README.md#inventory).

## Release Workflow

A single workflow (`docker-publish.yml`) is triggered by any `release/X.Y.Z` tag, or manually via
`workflow_dispatch`. **Every release publishes every container at `X.Y.Z`**, together with the Python
`.whl` and `sboms.zip` on the GitHub Release. There is one release cycle: the library and the containers
it runs always ship together. The version must be `X.Y.Z` with an optional `rcN`, `aN` or `bN` suffix,
which marks a prerelease; anything else fails the run before building.

Three jobs: `build` validates the tag, builds the wheel with `task build` on the runner (the version
injected into `src/arduino/version.py`, the project plus its `build` dependency group installed by uv),
then bakes and pushes every image; `sbom` scans the published images; `publish` assembles `sboms.zip`
and creates the GitHub Release with the wheel attached.

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

## Build Graph

The base image of every container is declared exactly once, in the `FROM` of the final stage of its
Dockerfile. A container deriving from another container of this repository writes it as
`FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION}`, and its `docker-bake.hcl` target links the
same parent with `parent_context()`. Bake then builds the parent in-graph before the child, however deep
the chain, with a single invocation and no hardcoded ordering: a release is `docker buildx bake --push`
over the `default` group, which lists every container.

`scripts/container_deps.py` reads the same `FROM` lines to serve everything else that needs the graph:
the dev workflow widens its selection with it, `scripts/sbom_delta.py` takes the base image to diff
against from it, and `task containers:tree` prints the hierarchy. The release checks that the
Dockerfiles and the bake targets describe the same set of containers before building.

## Adding a New Container

1. Create `containers/<group>/my-container/Dockerfile`, filing it under the group that describes what it
   is for — see [Layout](#layout). To derive from another container of this repo, start it with:

```dockerfile
ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}
```

2. Add a target to `docker-bake.hcl` and list it in the `default` group:

```hcl
target "my-container" {
  inherits   = ["_downstream"]              # "_common" when the base image is external
  context    = "containers/<group>/my-container"
  tags       = image_tags("my-container")
  cache-from = cache_from("my-container")
  cache-to   = cache_to("my-container")
  contexts   = parent_context("python-slim") # only when deriving from a container of this repo
  args       = { SOME_URL = "...", SOME_DIGEST = "sha256:..." }
}
```

3. If the image installs Python packages, declare them in a `pyproject.toml` locked by `task deps:lock`
   and register the container in the dependency license scan, see
   [scripts/licensed/README.md](../scripts/licensed/README.md).
4. Push a `release/X.Y.Z` tag — the workflow builds and publishes every target of the `default` group.

Check the result with `docker buildx bake --print my-container` and `task containers:tree`.

## docker-bake.hcl Reference

Variables the workflows set, all optional for local builds:

| Variable | Default | Description |
|---|---|---|
| `REGISTRY` | `ghcr.io/arduino/` | Registry prefix the images are published under, with a trailing slash |
| `IMAGE_TAG` | `local` | Tag applied to every built image: the release version or the dev tag |
| `RUN_NUMBER` | empty | When set, images are additionally tagged `<IMAGE_TAG>-<RUN_NUMBER>` (dev builds) |
| `TAG_LATEST` | `false` | When `true`, images are additionally tagged `latest` (non-prerelease releases) |
| `BASE_IMAGE_VERSION` | `local` | Tag the downstream `FROM` lines reference; parents are built in-graph, CI sets it to `IMAGE_TAG` |
| `CACHE_TAG` | empty | Tag of the per-image registry build cache, `buildcache` for releases and `<IMAGE_TAG>-buildcache` for dev builds; empty disables the cache |
| `SKIP_CACHE` | `false` | When `true`, the cache is not imported but still exported |

Two targets take extra named contexts: `python-apps-base` installs the wheel from `wheel` (`dist/`,
filled by `task build` with the wheel, `pyproject.toml` and `uv.lock`) and `models-downloader` reads
`models-list.yaml` from `models` (the repository's `models/` directory).

## SBOMs

`sboms.zip`, attached to every GitHub Release, is generated from the published images, so nothing SBOM-related lives in the tree. Every image is built by the release, so the archive covers all of them at the release version. `scripts/sbom_delta.py` scans one `name:version` with Syft against the base image of its Dockerfile, a parent container at the same version or the external image, and writes `<name>-<version>/{base,full,delta}.spdx.json`.

Scanning runs in the `_sbom-scan.yml` reusable workflow once the build is pushed, one matrix leg per image. Each leg uploads a `sbom-delta-<name>-<version>` artifact; a failed scan is a warning, never a failure. `publish` collects the artifacts, checks them against the released set, writes any gap to `MISSING.txt` inside the archive and to the job summary, and attaches the zip.

The dev workflow runs the same scan over the images it published when its `sbom` input is set, as `sbom-delta-<name>-<tag>` run artifacts.

## Dev Build Workflow

`docker-build.yml` ("DEV - Build & Publish Branch Containers") is triggered manually via `workflow_dispatch` with:

- `branch` — branch to build (defaults to the branch the workflow is run from)
- `containers` — comma-separated list of containers to build, or `all` (default)
- `tag` — optional custom image tag
- `skip_cache` — rebuild without cache
- `sbom` — also generate the SBOMs of the published images (off by default)

Images are tagged `dev-<branch-name>` (branch name lowercased and sanitized, e.g. `feat/My-Feature` → `dev-feat-my-feature`), plus a run-number-suffixed alias (e.g. `dev-feat-my-feature-42`), unless a custom `tag` is provided.

**Dependency ordering**: `scripts/container_deps.py closure` widens the selection with the containers deriving from it and with its bases, so the published set stays consistent, then a single `docker buildx bake` builds the result in dependency order through the parent links of `docker-bake.hcl`. Nothing is hardcoded in the workflow.

**Wheel**: when a selected target has a `wheel` context, the wheel is built first on the runner with `BRICKS_RELEASE_VERSION=<image-tag>`, so the compose files it bundles reference the dev images of the same run.

## Image Cleanup

`docker-cleanup.yml` runs two independent jobs:

| Trigger | Job | What it does |
|---|---|---|
| Branch deletion | `cleanup` | Deletes every GHCR version tagged `dev-<deleted-branch>`, including the run-number aliases and the build cache |
| Weekly (Sunday 03:00 UTC) or manual | `prune-untagged` | Deletes untagged container versions, the blobs orphaned by overwritten buildx cache manifests. Any tagged version is preserved, and each candidate is re-checked right before deletion. The manual run accepts a `dry_run` flag to only list what would be deleted. |

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: Buildx registry cache per image, `<image>:buildcache` for releases and `<image>:<tag>-buildcache` for dev builds (`mode=max`), invalidated by content; `skip_cache` forces a cold rebuild
- **Release assets**: The `.whl` and `sboms.zip` are uploaded to the GitHub Release via `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`, measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR is found, it falls back to the GitHub Actions Job Summary.
