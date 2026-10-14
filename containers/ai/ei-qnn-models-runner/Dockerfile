# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

FROM public.ecr.aws/g7a8t7v6/inference-container-qnn:v1.95.14@sha256:5742959f4b414c997fec333a2fa8429aa2e3d3cf31abdc9ab821fb20ade07574

ARG DEBIAN_FRONTEND=noninteractive

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/
COPY --chown=arduino:arduino ./src/start-runner.sh /home/arduino/start-runner.sh
COPY ./pyproject.toml ./uv.lock /tmp/deps/

# uv is only mounted while installing, it is not part of the image
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    set -ex; \
    uv export --frozen --project /tmp/deps | uv pip install --system --break-system-packages --no-cache-dir --require-hashes -r -; \
    rm -rf /tmp/deps; \
    chmod +x /home/arduino/start-runner.sh

WORKDIR /home/arduino
ENV HOME=/home/arduino

USER arduino
ENV USER=arduino

ENTRYPOINT ["/home/arduino/start-runner.sh"]
