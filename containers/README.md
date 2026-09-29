# Containers

Every container image produced by this repo lives here, one directory per image.

## Layout

```
containers/
├── base/     shared base images — never released on their own
├── ai/       AI/ML model runners
└── bricks/   the library itself and its supporting tooling
```

The sub-folder is the **release unit**: a tag's prefix selects the folder to release, so which images a
tag ships is decided by the layout alone (see [Release process](#release-process)).

Apart from that, the group is not part of a container's identity. A container is always referred to by
its **leaf directory name**, which is also its image name — `ghcr.io/arduino/app-bricks/<name>` — and
the value used in `downstream`, in the CI build matrices and in the `containers` input of the dev
workflow. CI finds a container by globbing `containers/*/<name>/ci.json`, so names must be unique across
groups; the build planner fails if two groups declare the same one.

## Inventory

| Container | Group | Built `FROM` | Purpose |
|---|---|---|---|
| `python-slim` | base | `python:3.13-slim-trixie` | Minimal Python layer shared by everything else |
| `python-base` | base | `python-slim` | System deps, non-root user, fonts, OpenCV wheel, libcamera + GStreamer packages |
| `qairt-common-base` | base | `python:3.13-slim-trixie` | Qualcomm AI Runtime SDK and a source build of FastRPC, shared by the LiteRT NPU runners |
| `aihub-framework` | base | `python-slim` | Source-only: the `aihub` runner framework and app skeleton, shared by the two runner bases |
| `python-apps-base` | bricks | `python-base` | App runtime: installs the Arduino App Bricks `.whl` and the Streamlit config |
| `models-downloader` | bricks | `python-slim` | Downloads models from AI Hub, Edge Impulse and Hugging Face per `models/models-list.yaml` |
| `aihub-litert-models-runner` | ai | `qairt-common-base` | Runs Qualcomm AI Hub models on LiteRT, with GStreamer/WebSocket input and MJPEG/WebSocket output. Installs OpenCV and `ai-edge-litert` for its runners |
| `aihub-onnx-models-runner` | ai | `python-slim` | Same framework on ONNX Runtime: the QNN execution provider plus Debian's FastRPC libraries and OpenCV, without the QAIRT SDK |
| `gesture-recognition-runner` | ai | `aihub-litert-models-runner` | Hand-gesture recognition on the MediaPipe palm/landmark/classifier models |
| `pose-estimation-runner` | ai | `aihub-litert-models-runner` | Body pose estimation on the PoseNet MobileNet model, 17 keypoints per person, custom pose models supported |
| `ocr-runner` | ai | `aihub-onnx-models-runner` | EasyOCR text detection and recognition on the Hexagon NPU |
| `ei-models-runner` | ai | Edge Impulse inference image | Edge Impulse inference with the bundled out-of-the-box models |
| `ei-qnn-models-runner` | ai | Edge Impulse QNN inference image | Same, on the NPU-accelerated (QNN) models |
| `llamacpp-runner` | ai | `python-slim` | llama.cpp model router, CPU build |
| `llamacpp-npu-runner` | ai | `qairt-common-base` | llama.cpp model router, Hexagon NPU build |

```mermaid
graph LR
  slim[python-slim] --> base[python-base] --> apps[python-apps-base]
  slim --> dl[models-downloader]
  slim --> lcpp[llamacpp-runner]
  qairt[qairt-common-base] --> aihub[aihub-litert-models-runner] --> gesture[gesture-recognition-runner]
  aihub --> pose[pose-estimation-runner]
  qairt --> lcppnpu[llamacpp-npu-runner]
  ei[ei-models-runner]
  eiqnn[ei-qnn-models-runner]
```

Solid arrows are `FROM`; the dotted ones are `COPY --from`: `aihub-framework` is a
source-only image that exists so the two runner bases share one copy of the `aihub`
package. CI builds each container with its own directory as the build context, so sharing
a plain directory between them is not possible - both pull the files out of that image
instead. It is still an ordinary edge in the graph: the parent lists both runners in its
`downstream`, and `tests/containers/test_container_graph.py` checks that every in-repo image
a Dockerfile pulls, `FROM` or `COPY --from`, is declared that way.

`ei-models-runner` and `ei-qnn-models-runner` build on external Edge Impulse images and have no upstream
inside this repo.

## Anatomy of a container directory

| Path | Required | Description |
|---|---|---|
| `Dockerfile` | yes | Build recipe. The directory itself is the build context. |
| `ci.json` | yes | CI metadata: watched paths, build args, dependencies, release flags |
| `requirements.txt` | if Python packages are installed | The Python packages the image installs. Never install them inline, the [dependency license scan](../scripts/licensed/README.md) only sees this file |

SBOMs are not kept in the tree: they are generated from the published images at release time (see
[SBOMs](#sboms)) and by the dev workflow as run artifacts.

An image that derives from another container in this repo must declare `ARG REGISTRY` and
`ARG BASE_IMAGE_VERSION` and use them in its `FROM`, so CI can point it at the freshly built upstream
instead of `latest`. Its parent must list it in `downstream`, and its own `sbom.runtime_base` must match
its `FROM`.

Note there is no `tag_prefix` in `ci.json` — the directory decides which tag releases the image. See the
[ci.json reference](../.github/README.md#cijson-reference) for every field.

## Release process

The tag prefix is the folder to release:

| Tag | Releases | Extra |
|---|---|---|
| `ai/X.Y.Z` | everything in `containers/ai/` | Opens a PR updating the compose files that reference the runners |
| `bricks/X.Y.Z` | everything in `containers/bricks/` | Builds the Python `.whl` and attaches it, plus the SBOMs of every distributed image, to the GitHub Release |

Pushing the tag runs `docker-publish.yml`, which:

1. **Resolves the build set** — the tagged folder, plus everything that derives from it, plus every base
   image any of those need. Base images in `containers/base/` are therefore built and tagged with the
   release version, but tagging the `base` group alone releases nothing.
2. **Orders it into waves** — `level_0` are the images with no dependency being built in the same run,
   each later wave builds on the previous one. So `bricks/X.Y.Z` builds `python-slim`, then
   `python-base` and `models-downloader`, then `python-apps-base`. There are four waves
   (`MAX_LEVELS` in `scripts/build_levels.py`, matched by the `build-l0`..`build-l3` jobs in
   the workflows); the deepest chain today is `python-slim` → `aihub-framework` →
   `aihub-onnx-models-runner` → `ocr-runner`.
3. **Skips what has not changed** — for `level_0` only, if a container's `watch_paths` are untouched
   since the previous tag of its own group, the existing image is re-tagged with `crane copy` instead of
   rebuilt. Later waves always rebuild, since their base was just rebuilt.
4. **Publishes** to `ghcr.io/arduino/app-bricks/<name>:X.Y.Z`, adding `:latest` for the containers that
   set `tag_latest`.

A tag whose prefix is not an existing folder fails the run with the list of valid groups.

## SBOMs

The `bricks/X.Y.Z` release attaches `sboms.zip` to the GitHub Release, covering **every image we
distribute**: the containers built by that release at `X.Y.Z`, plus the runners of the other groups at
the version the compose files under `src/` pin them to, plus the base images of both sets. The set is
resolved by `scripts/distributed_images.py` from `ci.json` and the compose files, and each image is
scanned with `scripts/sbom_delta.py` against the base image it was built from (`sbom.runtime_base`).
Each wave is scanned as soon as it is pushed, while the next wave builds; the pinned images are scanned
from the start. The archive holds one `<name>-<version>/` folder per image with `base`, `full` and
`delta` SPDX documents. A failed scan never blocks the release: the image is reported as a warning and
listed in `MISSING.txt` inside the archive.

Since the compose references are what ties an ai release to the library, release `ai/*` first, merge
the bot PRs that bump those references, then release `bricks/*`.

> `bricks/*` replaced `release/*` as the library release prefix when the containers were grouped by
> folder. Older `release/*` tags remain readable by `setuptools_scm` but no longer trigger a release.

## Development builds

`docker-build.yml` is manual (`workflow_dispatch`): pick a branch, and either `all` or a comma-separated
list of container names. The same dependency resolution applies — selecting a leaf pulls in its bases,
selecting a base pulls in everything derived from it — and images are published as
`ghcr.io/arduino/app-bricks/<name>:dev-<branch>`. They are deleted automatically when the branch is.

Full CI documentation: [`.github/README.md`](../.github/README.md).
