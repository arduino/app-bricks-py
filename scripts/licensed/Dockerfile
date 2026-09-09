# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Runs licensed against one venv per app declared in .licensed.yml.
#
# Convention: every stage is named licensed-<something> and every RUN goes through a
# licensed-* helper from scripts/licensed/bin. BuildKit records the stage name for COPY
# steps and the command for RUN steps, so the word "licensed" ends up in every cache
# record of this image and `task license:deps:clean` can prune them with a single filter.
# Keep to it when adding stages, an inline RUN would escape that filter.

ARG LICENSED_VERSION=5.0.6
ARG LICENSEE_VERSION=9.20.0

# Base stage shared by the venv builders
FROM python:3.13-slim AS licensed-base
COPY --chmod=755 scripts/licensed/bin/ /usr/local/bin/
RUN licensed-apt build-essential pkg-config libasound2-dev portaudio19-dev

# Builder stage for the licensed gem, rugged needs to be compiled
FROM licensed-base AS licensed-builder
ARG LICENSED_VERSION
ARG LICENSEE_VERSION
RUN licensed-apt cmake ruby-full ruby-dev libssl-dev zlib1g-dev libzstd-dev
RUN licensed-gem "${LICENSED_VERSION}" "${LICENSEE_VERSION}"

# One cached venv stage per scanned app
FROM licensed-base AS licensed-venv-python-base
COPY containers/base/python-base/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv python-base -r /req.txt

FROM licensed-base AS licensed-venv-aihub-models-runner
COPY containers/ai/aihub-models-runner/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv aihub-models-runner -r /req.txt

FROM licensed-base AS licensed-venv-gesture-recognition-runner
COPY containers/ai/gesture-recognition-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv gesture-recognition-runner -r /req.txt

FROM licensed-base AS licensed-venv-pose-estimation-runner
COPY containers/ai/pose-estimation-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv pose-estimation-runner -r /req.txt

FROM licensed-base AS licensed-venv-models-downloader
COPY containers/bricks/models-downloader/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv models-downloader -r /req.txt

FROM licensed-base AS licensed-venv-ei-qnn-models-runner
COPY containers/ai/ei-qnn-models-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv ei-qnn-models-runner -r /req.txt

FROM licensed-base AS licensed-venv-llamacpp-runner
COPY containers/ai/llamacpp-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv llamacpp-runner -r /req.txt

FROM licensed-base AS licensed-venv-llamacpp-npu-runner
COPY containers/ai/llamacpp-npu-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip licensed-venv llamacpp-npu-runner -r /req.txt

# python-apps-base ships the library wheel with all extras, resolved here from the sources
FROM licensed-base AS licensed-venv-python-apps-base
COPY pyproject.toml .
COPY src/arduino/app_tools/ src/arduino/app_tools/
RUN --mount=type=cache,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 licensed-venv python-apps-base ".[all]"

# Final stage where licensed runs
FROM python:3.13-slim AS licensed

ENV LANG=C.UTF-8

COPY --chmod=755 scripts/licensed/bin/licensed-apt /usr/local/bin/
RUN licensed-apt git ruby libssl3 zlib1g libzstd1

COPY --from=licensed-builder /var/lib/gems /var/lib/gems
COPY --from=licensed-builder /usr/local/bin/licensed /usr/local/bin/licensed

COPY --from=licensed-venv-python-base                /venvs/python-base                  /venvs/python-base
COPY --from=licensed-venv-aihub-models-runner        /venvs/aihub-models-runner          /venvs/aihub-models-runner
COPY --from=licensed-venv-gesture-recognition-runner /venvs/gesture-recognition-runner   /venvs/gesture-recognition-runner
COPY --from=licensed-venv-pose-estimation-runner     /venvs/pose-estimation-runner       /venvs/pose-estimation-runner
COPY --from=licensed-venv-models-downloader          /venvs/models-downloader            /venvs/models-downloader
COPY --from=licensed-venv-ei-qnn-models-runner        /venvs/ei-qnn-models-runner /venvs/ei-qnn-models-runner
COPY --from=licensed-venv-llamacpp-runner             /venvs/llamacpp-runner     /venvs/llamacpp-runner
COPY --from=licensed-venv-llamacpp-npu-runner         /venvs/llamacpp-npu-runner /venvs/llamacpp-npu-runner
COPY --from=licensed-venv-python-apps-base           /venvs/python-apps-base             /venvs/python-apps-base

COPY --chmod=755 scripts/licensed/entrypoint.sh /entrypoint.sh

WORKDIR /src

ENTRYPOINT ["/entrypoint.sh"]
