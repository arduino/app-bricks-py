# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

# The compose files provisioned at runtime reference the images published with this one
ARG BASE_IMAGE_VERSION
ENV BRICKS_RELEASE_VERSION=${BASE_IMAGE_VERSION}

# Copy the Streamlit configuration file with customized settings in home app shared directory
COPY ./streamlit/config.toml /home/app/.streamlit/config.toml

COPY ./scripts/*.sh /
# The wheel and the lock it was built from, from the bake context filled by `task build:bricks`
COPY --from=wheel arduino*.whl pyproject.toml uv.lock /app/
COPY ./patches/ /tmp/patches/
# OpenCV is left out, python-base ships its own GStreamer build of it
RUN set -ex; \
    uv export --frozen --project /app --extra all --no-dev --no-emit-project --no-emit-package opencv-python-headless | uv pip install --system --no-cache-dir --require-hashes -r -; \
    uv pip install --system --no-cache-dir --no-deps /app/*.whl; \
    rm /app/*.whl /app/pyproject.toml /app/uv.lock; \
    # Every patch in patches/ (see its README), in name order: one that no longer applies exactly fails the build.
    # Applied before compileall, which compiles the patched sources. GNU patch is installed for this step only
    set -- /tmp/patches/*.patch; \
    if [ -e "$1" ]; then \
        apt-get update -y; \
        apt-get install -y --no-install-recommends --no-install-suggests patch; \
        for p in "$@"; do patch -p1 --forward --fuzz=0 --no-backup-if-mismatch -d /usr/local/lib/python3.13/site-packages -i "$p"; done; \
        apt-get purge -y --auto-remove patch; \
        apt-get dist-clean; \
        rm -rf /var/lib/apt/lists/* /var/lib/dpkg/*-old; \
    fi; \
    rm -rf /tmp/patches; \
    # ALSA dsnoop/dmix wrappers: the file is static, generate it once here instead of at every app start
    HOME=/home/app bash /provision-alsa-devices.sh; \    
    # fix permissions for /home/app
    chown arduino:arduino -R /home/app; \
    chmod 777 -R /home/app; \
    # ensure permissions for /run.sh
    chmod 755 /*.sh; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/home/app
ENV USER=arduino
ENV PIPEWIRE_REMOTE=/run/user/1000/pipewire-0

ENTRYPOINT [ "/run.sh" ]
