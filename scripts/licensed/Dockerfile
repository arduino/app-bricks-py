# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Tools image for the dependency license scan, see README.md. The venvs are built at run
# time by run.py into the /venvs volume, so this image knows nothing about the scanned apps.

ARG LICENSED_VERSION=5.0.6
ARG LICENSEE_VERSION=9.20.0

# Same base as the scanned containers, pinned so the venvs resolve like the images do
FROM python:3.13-slim-trixie@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS builder

ARG LICENSED_VERSION
ARG LICENSEE_VERSION

# rugged, a licensed dependency, needs to be compiled
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential cmake pkg-config ruby-full ruby-dev libssl-dev zlib1g-dev libzstd-dev; \
    gem install licensee -v "${LICENSEE_VERSION}"; \
    gem install licensed -v "${LICENSED_VERSION}"; \
    rm -rf /var/lib/gems/*/cache /var/lib/gems/*/gems/rugged-*/vendor /var/lib/gems/*/gems/rugged-*/ext

FROM python:3.13-slim-trixie@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV LANG=C.UTF-8 PIP_ROOT_USER_ACTION=ignore

# Compilers and headers serve the packages that ship no wheel, such as pyalsaaudio and PyAudio
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends git ruby libssl3 zlib1g libzstd1 build-essential pkg-config libasound2-dev portaudio19-dev; \
    rm -rf /var/lib/apt/lists/*; \
    pip install --no-cache-dir pyyaml==6.0.2; \
    git config --global --add safe.directory /src

COPY --from=builder /var/lib/gems /var/lib/gems
COPY --from=builder /usr/local/bin/licensed /usr/local/bin/licensed
COPY --chmod=755 run.py /run.py

WORKDIR /src

ENTRYPOINT ["/run.py"]
