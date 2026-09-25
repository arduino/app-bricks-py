# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

COPY --chown=arduino:arduino --chmod=755 ./src /app
COPY --chown=arduino:arduino --chmod=644 ./requirements.txt /app/requirements.txt

# Runtime settings expected from the compose file: network_mode host to see the wireless
# interface, cap_drop ALL plus cap_add NET_ADMIN, read_only rootfs, no ports.
RUN set -ex; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --no-install-suggests iw libcap2-bin; \
    python -m pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    # Only iw carries CAP_NET_ADMIN, as a file capability, so the service itself runs unprivileged
    IW="$(readlink -f "$(command -v iw)")"; \
    setcap cap_net_admin+ep "$IW"; \
    getcap "$IW" | grep -q cap_net_admin; \
    apt-get purge -y --auto-remove libcap2-bin; \
    rm -rf /var/lib/apt/lists/*; \
    # Remove every setuid/setgid escalation path from the image
    find / -xdev -type f -perm /6000 -exec chmod a-s {} +; \
    # Mount point of the socket directory shared with the app container
    mkdir -p /app/.cache/tps_location_api; \
    chown -R arduino:arduino /app; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall -q /usr/local/lib /app

USER arduino

WORKDIR /app

ENV HOME=/app
ENV USER=arduino
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/start-scan-server.sh"]
