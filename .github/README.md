# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/<group>/<name>/`. Each container is described by a `ci.json` file in its directory that drives CI behaviour — no workflow changes are needed when adding a new container.

### Layout

Containers are filed under three groups. **The group is the release unit**: it is what a release tag
selects, so which images a tag ships is decided by the folder alone.

| Group | Released by | Contains |
|---|---|---|
| `containers/ai/` | `ai/X.Y.Z` | AI/ML model runners |
| `containers/bricks/` | `bricks/X.Y.Z` | Images shipping the library itself and its supporting tooling |
| `containers/base/` | never directly | Shared base images other containers derive `FROM` (`base_image: true`) |

Apart from selecting the release, the group is **not** part of a container's identity: a container is
always referred to by its leaf directory name, which is also its image name
(`ghcr.io/arduino/app-bricks/<name>`) and the value used in `downstream`, in the build matrices and in
the `containers` input of the dev workflow. CI locates a container by globbing
`containers/*/<name>/ci.json`, so moving a container between groups only means updating its
`watch_paths` — and changes which tag releases it. Leaf names must stay unique across groups; the build
planner fails loudly if two groups declare the same name.

| Image | Group | Base | Purpose |
|---|---|---|---|
| **python-slim** | `base` | `python:3.13-slim-trixie` | Minimal Python layer shared by every image |
| **python-base** | `base` | `python-slim` | Foundation layer — system deps, user/group setup, fonts |
| **qairt-common-base** | `base` | `python:3.13-slim-trixie` | Qualcomm AI Runtime deps shared by the NPU runners |
| **python-apps-base** | `bricks` | `python-base` | App runtime — installs the Arduino App Bricks `.whl`, Streamlit config |
| **models-downloader** | `bricks` | `python-slim` | Fetches the models declared in `models/models-list.yaml` |
| **ei-models-runner** | `ai` | Edge Impulse inference image | AI/ML model inference with OOTB models |

## Release Triggers (Tag-Based)

A single workflow (`docker-publish.yml`) handles all container releases. **The tag prefix is the
`containers/` group to release**: pushing `ai/0.12.0` releases every container under `containers/ai/`,
pushing `bricks/0.12.0` releases every container under `containers/bricks/`. Nothing in `ci.json`
decides this — only the folder does.

| Tag pattern | Containers | Extra behaviour |
|---|---|---|
| `bricks/X.Y.Z` | everything in `containers/bricks/` | Builds and uploads `.whl` to GitHub Release (displayed as `X.Y.Z`) |
| `ai/X.Y.Z` | everything in `containers/ai/` | Auto-creates a PR to update compose file references |

Containers flagged `base_image` are excluded from the seed set, so tagging the `base` group builds
nothing: base images are rebuilt (and tagged with the triggering release version) only as a dependency
of an image being released. A tag whose prefix is not an existing group fails the `detect` job with the
list of known groups, rather than silently building nothing.

> **Note**: `bricks/*` replaced `release/*` as the library release prefix when the containers were
> grouped by folder. `release/*` is still accepted by `tag_regex` in `pyproject.toml` so the pre-reorg
> tag history stays parseable for dev builds, but it no longer triggers a release.

## Dependency Ordering (Release)

The set of containers to build is the tagged group **plus the transitive closure in both directions**:
all descendants (so images derived from something being released are rebuilt) and all ancestors of that
set (so everything being rebuilt sits on a freshly built base). The result is split into topological
waves by `scripts/build_levels.py` — `level_0` has no in-set parent, each later wave depends on the
previous one — and `docker-publish.yml` chains one `build-l<n>` job per wave with `needs`.

So `bricks/X.Y.Z` builds `python-slim` → (`python-base`, `models-downloader`) → `python-apps-base`, in
that order, even though only the last two live in `containers/bricks/`. Dependencies are declared by the
`downstream` field in `ci.json`, which must mirror the `FROM` lines of the Dockerfiles.

Only the `level_0` wave may skip its build (see [Skip-Rebuild Logic](#skip-rebuild-logic)); later waves
always rebuild, since their base image was just rebuilt.

## Adding a New Container

1. Create `containers/<group>/my-container/Dockerfile`. The group decides which tag releases the image,
   so pick it accordingly — see [Layout](#layout).
2. Create `containers/<group>/my-container/ci.json`:

```json
{
  "watch_paths": ["containers/<group>/my-container/"],
  "tag_latest": false,
  "build_whl": false,
  "update_compose": false,
  "build_args": {},
  "sbom": { "runtime_base": "${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}" },
  "downstream": []
}
```

3. Push a tag `<group>/X.Y.Z` — the workflow picks up everything in that folder automatically.

To declare that another container depends on yours, add it to `downstream`:

```json
"downstream": ["my-other-container"]
```

> **Note**: any container listed in `downstream` must declare `ARG BASE_IMAGE_VERSION` in its Dockerfile. The CI passes the upstream image's tag via this build arg so the downstream image pulls the freshly built version, not `latest`.

No workflow file changes required.

## ci.json Reference

There is no `tag_prefix` field: the container's directory decides which tag releases it.

| Field | Type | Description |
|---|---|---|
| `watch_paths` | string[] | Repo-relative paths checked by the skip-rebuild logic — must include the container's own directory |
| `base_image` | bool | Shared base image: never a direct release target, only rebuilt as a dependency |
| `tag_latest` | bool | Also push a `:latest` tag on release |
| `build_whl` | bool | Build and upload the Python `.whl` before the Docker build |
| `update_compose` | bool | After release, open a PR updating `brick_compose.yaml` references |
| `build_args` | object | Docker build args passed to the Dockerfile (key/value pairs) |
| `sbom.runtime_base` | string | Image the delta SBOM is computed against — must match the Dockerfile's `FROM` |
| `downstream` | string[] | Containers that depend on this one — rebuilt automatically after this container is built |

## Skip-Rebuild Logic

For `level_0` containers only, the release checks whether the container's `watch_paths` actually changed since the previous tag of its own group:

- **Changed** → full Docker build and push
- **Unchanged** → `crane copy` re-tags the existing image to the new version (instant, no rebuild)

So releasing `bricks/X.Y.Z` when nothing under `containers/base/python-slim/` changed re-tags `python-slim` from the previous release instead of rebuilding it.

## Dev Build Workflow

`docker-github-build.yml` triggers on every push to non-`main` branches and builds only the containers whose `watch_paths` changed (detected via `git diff` against the previous commit). Images are tagged with the sanitized branch name (e.g. `feat/my-feature` → `feat-my-feature`) plus a run-number suffix (e.g. `feat-my-feature-42`).

**Dependency ordering**: the detect job splits containers into two groups:

- **Base containers** (`build` job): containers with no upstream being built in the same run — these build in parallel.
- **Downstream containers** (`build-downstream` job): containers whose upstream is also being built — these wait for the `build` job to complete before starting, and receive `BASE_IMAGE_VERSION=<branch-tag>` as a build arg so they use the freshly built upstream image.

The grouping is driven entirely by the `downstream` field in ci.json — no hardcoded container names in the workflow.

Can also be triggered manually via `workflow_dispatch` with:
- `containers` — comma-separated list of containers to build, or `all`
- `tag` — optional custom image tag

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: GitHub Actions cache (`type=gha`, `mode=max`)
- **Release assets**: A `bricks/*` release also uploads the `.whl` to the GitHub Release via `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`, measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR is found, it falls back to the GitHub Actions Job Summary.
