# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# syntax=docker/dockerfile:1

# ── Shared base for building venvs ────────────────────────────────────────────
FROM python:3.13-slim AS venv-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Independent venv stages (change one, only that stage rebuilds) ────────────
FROM venv-base AS venv-python-base
COPY containers/python-base/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/python-base && \
    /venvs/python-base/bin/pip install -q -r /req.txt

FROM venv-base AS venv-aihub-models-runner
COPY containers/aihub-models-runner/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/aihub-models-runner && \
    /venvs/aihub-models-runner/bin/pip install -q -r /req.txt

FROM venv-base AS venv-gesture-recognition-runner
COPY containers/gesture-recognition-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/gesture-recognition-runner && \
    /venvs/gesture-recognition-runner/bin/pip install -q -r /req.txt

FROM venv-base AS venv-arduino-app-bricks
COPY pyproject.toml .
COPY src/arduino/app_tools/ src/arduino/app_tools/
# RUN mkdir -p src/arduino/app_bricks && \
#     touch src/arduino/__init__.py src/arduino/app_bricks/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/arduino-app-bricks && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 /venvs/arduino-app-bricks/bin/pip install -q ".[all]"

# ── Final image ───────────────────────────────────────────────────────────────
FROM venv-base

ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    ruby-full \
    ruby-dev \
    libssl-dev \
    zlib1g-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN gem install licensed -v 5.0.6

COPY --from=venv-python-base                /venvs/python-base                  /venvs/python-base
COPY --from=venv-aihub-models-runner        /venvs/aihub-models-runner          /venvs/aihub-models-runner
COPY --from=venv-gesture-recognition-runner /venvs/gesture-recognition-runner   /venvs/gesture-recognition-runner
COPY --from=venv-arduino-app-bricks         /venvs/arduino-app-bricks           /venvs/arduino-app-bricks

COPY containers/licensed/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /src

ENTRYPOINT ["/entrypoint.sh"]
