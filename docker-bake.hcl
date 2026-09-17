# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Build definition of every container under containers/<group>/<name>/.
#
# A container deriving from another container of this repository declares it
# in its Dockerfile (FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION})
# and links it here with parent_context(): bake then builds the parent in-graph,
# in dependency order, however deep the chain. scripts/container_deps.py reads
# the same FROM lines and the release checks that the two agree. Targets are
# listed parents first, each followed by the containers deriving from it.
#
# Build arguments specific to a container (download URLs and digests) are the
# ARG defaults of its Dockerfile, so it also builds standalone.
#
#   docker buildx bake --print                  # inspect the resolved definition
#   docker buildx bake python-apps-base         # build a container and its parents
#   docker buildx bake                          # build them all
#
# python-apps-base installs the wheel from the "wheel" context (dist/, filled by
# `task build`), models-downloader reads models-list.yaml from the "models"
# context (models/).

# Registry prefix the images are published under, with a trailing slash.
variable "REGISTRY" {
  default = "ghcr.io/arduino/"
}

# Tag applied to the built images. CI sets the dev or release tag.
variable "IMAGE_TAG" {
  default = "local"
}

# When set, images are additionally tagged "${IMAGE_TAG}-${RUN_NUMBER}".
variable "RUN_NUMBER" {
  default = ""
}

# When "true", images are additionally tagged "latest".
variable "TAG_LATEST" {
  default = "false"
}

# Tag the downstream Dockerfiles' FROM lines reference. Parents are built
# in-graph through the parent links, so any value works; CI sets it to the
# published tag for consistency.
variable "BASE_IMAGE_VERSION" {
  default = "local"
}

# Tag holding the registry build cache. Empty disables cache import and export.
variable "CACHE_TAG" {
  default = ""
}

# Rebuild without importing the cache. The cache is still exported.
variable "SKIP_CACHE" {
  default = "false"
}

# Set by GitHub Actions, used for the OCI labels.
variable "GITHUB_REPOSITORY" {
  default = "arduino/app-bricks-py"
}

variable "GITHUB_SHA" {
  default = ""
}

function "image" {
  params = [container]
  result = "${REGISTRY}app-bricks/${container}"
}

function "image_tags" {
  params = [container]
  result = compact([
    "${image(container)}:${IMAGE_TAG}",
    RUN_NUMBER == "" ? "" : "${image(container)}:${IMAGE_TAG}-${RUN_NUMBER}",
    TAG_LATEST == "true" ? "${image(container)}:latest" : "",
  ])
}

function "cache_from" {
  params = [container]
  result = SKIP_CACHE == "true" || CACHE_TAG == "" ? [] : ["type=registry,ref=${image(container)}:${CACHE_TAG}"]
}

function "cache_to" {
  params = [container]
  result = CACHE_TAG == "" ? [] : ["type=registry,ref=${image(container)}:${CACHE_TAG},mode=max"]
}

# Resolves the parent image reference of a downstream container's FROM line to
# the parent's bake target, so bake builds it in-graph in dependency order.
function "parent_context" {
  params = [parent]
  result = { "${image(parent)}:${BASE_IMAGE_VERSION}" = "target:${parent}" }
}

# Every image carries the SBOM BuildKit generates while building it, read with
# `docker buildx imagetools inspect <image> --format '{{ json .SBOM }}'`.
target "_common" {
  platforms = ["linux/arm64"]
  attest    = ["type=provenance,disabled=true", "type=sbom"]
  labels = {
    "org.opencontainers.image.source"   = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.url"      = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.revision" = GITHUB_SHA
    "org.opencontainers.image.version"  = IMAGE_TAG
  }
}

# Containers whose Dockerfile starts from another container of this repository.
target "_downstream" {
  inherits = ["_common"]
  args = {
    REGISTRY           = REGISTRY
    BASE_IMAGE_VERSION = BASE_IMAGE_VERSION
  }
}

# Every container: a release builds and publishes them all.
group "default" {
  targets = [
    "python-slim",
    "llamacpp-runner",
    "models-downloader",
    "python-base",
    "python-apps-base",
    "tps-location-api",
    "qairt-common-base",
    "aihub-models-runner",
    "gesture-recognition-runner",
    "pose-estimation-runner",
    "llamacpp-npu-runner",
    "ei-models-runner",
    "ei-qnn-models-runner",
  ]
}

target "python-slim" {
  inherits   = ["_common"]
  context    = "containers/base/python-slim"
  tags       = image_tags("python-slim")
  cache-from = cache_from("python-slim")
  cache-to   = cache_to("python-slim")
}

target "llamacpp-runner" {
  inherits   = ["_downstream"]
  context    = "containers/ai/llamacpp-runner"
  tags       = image_tags("llamacpp-runner")
  cache-from = cache_from("llamacpp-runner")
  cache-to   = cache_to("llamacpp-runner")
  contexts   = parent_context("python-slim")
}

target "models-downloader" {
  inherits   = ["_downstream"]
  context    = "containers/bricks/models-downloader"
  tags       = image_tags("models-downloader")
  cache-from = cache_from("models-downloader")
  cache-to   = cache_to("models-downloader")
  contexts = merge(
    { models = "models" },
    parent_context("python-slim"),
  )
}

target "python-base" {
  inherits   = ["_downstream"]
  context    = "containers/base/python-base"
  tags       = image_tags("python-base")
  cache-from = cache_from("python-base")
  cache-to   = cache_to("python-base")
  contexts   = parent_context("python-slim")
}

target "python-apps-base" {
  inherits   = ["_downstream"]
  context    = "containers/bricks/python-apps-base"
  tags       = image_tags("python-apps-base")
  cache-from = cache_from("python-apps-base")
  cache-to   = cache_to("python-apps-base")
  contexts = merge(
    { wheel = "dist" },
    parent_context("python-base"),
  )
}

target "tps-location-api" {
  inherits   = ["_downstream"]
  context    = "containers/bricks/tps-location-api"
  tags       = image_tags("tps-location-api")
  cache-from = cache_from("tps-location-api")
  cache-to   = cache_to("tps-location-api")
  contexts   = parent_context("python-slim")
}

target "qairt-common-base" {
  inherits   = ["_common"]
  context    = "containers/base/qairt-common-base"
  tags       = image_tags("qairt-common-base")
  cache-from = cache_from("qairt-common-base")
  cache-to   = cache_to("qairt-common-base")
}

target "aihub-models-runner" {
  inherits   = ["_downstream"]
  context    = "containers/ai/aihub-models-runner"
  tags       = image_tags("aihub-models-runner")
  cache-from = cache_from("aihub-models-runner")
  cache-to   = cache_to("aihub-models-runner")
  contexts   = parent_context("qairt-common-base")
}

target "gesture-recognition-runner" {
  inherits   = ["_downstream"]
  context    = "containers/ai/gesture-recognition-runner"
  tags       = image_tags("gesture-recognition-runner")
  cache-from = cache_from("gesture-recognition-runner")
  cache-to   = cache_to("gesture-recognition-runner")
  contexts   = parent_context("aihub-models-runner")
}

target "pose-estimation-runner" {
  inherits   = ["_downstream"]
  context    = "containers/ai/pose-estimation-runner"
  tags       = image_tags("pose-estimation-runner")
  cache-from = cache_from("pose-estimation-runner")
  cache-to   = cache_to("pose-estimation-runner")
  contexts   = parent_context("aihub-models-runner")
}

target "llamacpp-npu-runner" {
  inherits   = ["_downstream"]
  context    = "containers/ai/llamacpp-npu-runner"
  tags       = image_tags("llamacpp-npu-runner")
  cache-from = cache_from("llamacpp-npu-runner")
  cache-to   = cache_to("llamacpp-npu-runner")
  contexts   = parent_context("qairt-common-base")
}

target "ei-models-runner" {
  inherits   = ["_common"]
  context    = "containers/ai/ei-models-runner"
  tags       = image_tags("ei-models-runner")
  cache-from = cache_from("ei-models-runner")
  cache-to   = cache_to("ei-models-runner")
}

target "ei-qnn-models-runner" {
  inherits   = ["_common"]
  context    = "containers/ai/ei-qnn-models-runner"
  tags       = image_tags("ei-qnn-models-runner")
  cache-from = cache_from("ei-qnn-models-runner")
  cache-to   = cache_to("ei-qnn-models-runner")
}
