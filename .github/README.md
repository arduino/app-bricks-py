# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/`. Each container is described by a `ci.json` file in its directory that drives CI behaviour — no workflow changes are needed when adding a new container.

| Image | Base | Purpose |
|---|---|---|
| **python-base** | `python:3.13-slim` | Foundation layer — system deps, user/group setup, fonts |
| **python-apps-base** | `python-base` | App runtime — installs the Arduino App Bricks `.whl`, Streamlit config |
| **ei-models-runner** | Edge Impulse inference image | AI/ML model inference with OOTB models |

## Release Triggers (Tag-Based)

A single workflow (`docker-github-publish.yml`) handles all container releases. It is triggered by any `prefix/X.Y.Z` tag. The prefix is matched against each container's `ci.json` to determine what to build.

| Tag pattern | Container | Extra behaviour |
|---|---|---|
| `base/X.Y.Z` | `python-base:X.Y.Z` + `:latest` | Automatically triggers a rebuild of downstream containers |
| `release/X.Y.Z` | `python-apps-base:X.Y.Z` | Builds and uploads `.whl` to GitHub Release (displayed as `X.Y.Z`) |
| `ai/X.Y.Z` | `ei-models-runner:X.Y.Z` | Auto-creates a PR to update compose file references |

If the pushed tag prefix does not match any container's `ci.json`, the workflow exits cleanly with no build.

## Downstream Cascade (Release)

When a container is built or retagged, the workflow checks its `downstream` field in `ci.json`. For each downstream container listed, it finds the latest tag for that container and automatically dispatches a new release workflow run. This ensures dependent images are always rebuilt after their base image changes.

For example, pushing `base/X.Y.Z` builds `python-base` and then automatically triggers a new run for the latest `release/X.Y.Z` tag, rebuilding `python-apps-base` on top of the new base.

The dispatch only happens after the upstream build completes successfully.

## Adding a New Container

1. Create `containers/my-container/Dockerfile`
2. Create `containers/my-container/ci.json`:

```json
{
  "tag_prefix": "my-prefix",
  "watch_paths": ["containers/my-container/"],
  "tag_latest": false,
  "build_whl": false,
  "update_compose": false,
  "build_args": {},
  "downstream": []
}
```

3. Push a tag `my-prefix/X.Y.Z` — the workflow picks it up automatically.

To declare that another container depends on yours, add it to `downstream`:

```json
"downstream": ["my-other-container"]
```

> **Note**: any container listed in `downstream` must declare `ARG BASE_IMAGE_VERSION` in its Dockerfile. The CI passes the upstream image's tag via this build arg so the downstream image pulls the freshly built version, not `latest`.

No workflow file changes required.

## ci.json Reference

| Field | Type | Description |
|---|---|---|
| `tag_prefix` | string | Tag namespace that triggers this container's release (e.g. `release`) |
| `watch_paths` | string[] | Paths checked by the skip-rebuild logic and dev build change detection |
| `tag_latest` | bool | Also push a `:latest` tag on release |
| `build_whl` | bool | Build and upload the Python `.whl` before the Docker build |
| `update_compose` | bool | After release, open a PR updating `brick_compose.yaml` references |
| `build_args` | object | Docker build args passed to the Dockerfile (key/value pairs) |
| `downstream` | string[] | Containers that depend on this one — rebuilt automatically after this container is built |

## Skip-Rebuild Logic

Every release checks whether the container's source files actually changed since the previous tag of the same prefix:

- **Changed** → full Docker build and push
- **Unchanged** → `crane copy` re-tags the existing image to the new version (instant, no rebuild)

This means releasing a new `release/X.Y.Z` when only `ei-models-runner` sources changed will re-tag `python-apps-base` without rebuilding it.

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
- **Release assets**: The `release/*` workflow also uploads the `.whl` to the GitHub Release via `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`, measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR is found, it falls back to the GitHub Actions Job Summary.
