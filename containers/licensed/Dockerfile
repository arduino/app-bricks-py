# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# Base stage shared for building venvs and the licensed gem
FROM python:3.13-slim AS venv-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Builder stage for the licensed gem (rugged needs to be compiled)
FROM venv-base AS licensed-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ruby-full \
    ruby-dev \
    libssl-dev \
    zlib1g-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
RUN gem install licensed -v 5.0.6 && \
    rm -rf /var/lib/gems/*/cache \
           /var/lib/gems/*/gems/rugged-*/vendor \
           /var/lib/gems/*/gems/rugged-*/ext

# Cached venv stages
FROM venv-base AS venv-python-base
COPY containers/python-base/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/python-base && \
    /venvs/python-base/bin/pip install -q -r /req.txt && \
    find /venvs/python-base/lib/python*/site-packages -mindepth 1 -maxdepth 1 \
        ! -name '*.dist-info' \
        ! -name 'pip' ! -name 'setuptools' ! -name 'wheel' \
        ! -name 'pkg_resources' ! -name '_distutils_hack' \
        -exec rm -rf {} +

FROM venv-base AS venv-aihub-models-runner
COPY containers/aihub-models-runner/requirements-base.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/aihub-models-runner && \
    /venvs/aihub-models-runner/bin/pip install -q -r /req.txt && \
    find /venvs/aihub-models-runner/lib/python*/site-packages -mindepth 1 -maxdepth 1 \
        ! -name '*.dist-info' \
        ! -name 'pip' ! -name 'setuptools' ! -name 'wheel' \
        ! -name 'pkg_resources' ! -name '_distutils_hack' \
        -exec rm -rf {} +

FROM venv-base AS venv-gesture-recognition-runner
COPY containers/gesture-recognition-runner/requirements.txt /req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/gesture-recognition-runner && \
    /venvs/gesture-recognition-runner/bin/pip install -q -r /req.txt && \
    find /venvs/gesture-recognition-runner/lib/python*/site-packages -mindepth 1 -maxdepth 1 \
        ! -name '*.dist-info' \
        ! -name 'pip' ! -name 'setuptools' ! -name 'wheel' \
        ! -name 'pkg_resources' ! -name '_distutils_hack' \
        -exec rm -rf {} +

FROM venv-base AS venv-arduino-app-bricks
COPY pyproject.toml .
COPY src/arduino/app_tools/ src/arduino/app_tools/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /venvs/arduino-app-bricks && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 /venvs/arduino-app-bricks/bin/pip install -q ".[all]" && \
    find /venvs/arduino-app-bricks/lib/python*/site-packages -mindepth 1 -maxdepth 1 \
        ! -name '*.dist-info' \
        ! -name 'pip' ! -name 'setuptools' ! -name 'wheel' \
        ! -name 'pkg_resources' ! -name '_distutils_hack' \
        -exec rm -rf {} +

# Final stage where licensed runs
FROM python:3.13-slim

ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ruby \
    libssl3 \
    zlib1g \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=licensed-builder /var/lib/gems /var/lib/gems
COPY --from=licensed-builder /usr/local/bin/licensed /usr/local/bin/licensed

COPY --from=venv-python-base                /venvs/python-base                  /venvs/python-base
COPY --from=venv-aihub-models-runner        /venvs/aihub-models-runner          /venvs/aihub-models-runner
COPY --from=venv-gesture-recognition-runner /venvs/gesture-recognition-runner   /venvs/gesture-recognition-runner
COPY --from=venv-arduino-app-bricks         /venvs/arduino-app-bricks           /venvs/arduino-app-bricks

COPY containers/licensed/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /src

ENTRYPOINT ["/entrypoint.sh"]
