# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Base image for the AI runners that reach the Hexagon NPU through ONNX Runtime's QNN
# execution provider, as opposed to aihub-litert-models-runner, which reaches it
# through the LiteRT QNN delegate.
#
# It deliberately does not build on qairt-common-base. That image downloads and installs
# the whole QAIRT SDK (the aarch64 runtime plus the unsigned Hexagon skels for every HTP
# architecture, well over a GB) and builds fastrpc from source, and an ONNX runner uses
# none of it: the `onnxruntime-qnn` plugin wheel bundles its own QAIRT - backend library
# and skels - and that is the one the EP loads (see utils/onnx_ep.py in a runner). What is
# actually needed from the system is only the FastRPC client libraries that carry the
# calls to the DSP, and Debian ships those as a package.

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

# The `aihub` framework and the app skeleton are shared with aihub-litert-models-runner
# and live in their own source-only image; see containers/base/aihub-framework.
FROM ${REGISTRY}app-bricks/aihub-framework:${BASE_IMAGE_VERSION} AS framework

FROM ${REGISTRY}app-bricks/python-slim:${BASE_IMAGE_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN set -ex; \
    # The Qualcomm FastRPC libraries live in contrib, and for trixie only in
    # backports: enable just that component, for this build only.
    printf '%s\n' 'Types: deb' 'URIs: http://deb.debian.org/debian' \
        'Suites: trixie-backports' 'Components: contrib' \
        'Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp' \
        > /etc/apt/sources.list.d/fastrpc-backports.sources; \
    apt-get update -y; \
    # netcat-openbsd: the brick compose files probe the runner's websocket port
    # with `nc -z` as their healthcheck.
    apt-get install -y --no-install-recommends --no-install-suggests netcat-openbsd; \
    # Install the FastRPC client libraries used to reach the Hexagon DSP:
    # libcdsprpc.so.1, libadsprpc.so.1, libsdsprpc.so.1 and their listeners.
    # This Debian build keeps the upstream paths: it reads the machine name from
    # /sys/firmware/devicetree/base/model (no /run/device-model bind mount) and
    # the DSP config from /usr/share/hexagon-dsp/conf.d, not /usr/share/qcom.
    apt-get install -y --no-install-recommends --no-install-suggests -t trixie-backports libfastrpc1; \
    rm /etc/apt/sources.list.d/fastrpc-backports.sources; \
    # libfastrpc1 ships only the SONAME symlinks: add the unversioned ones,
    # which live in libfastrpc-dev, so dlopen("libcdsprpc.so") resolves too.
    for so in /usr/lib/aarch64-linux-gnu/lib*dsp*.so.1; do ln -sf "${so##*/}" "${so%.so.1}.so"; done; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    apt-get dist-clean; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/cache/apt/*.bin; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    ldconfig

# Baked fastrpc DSP path config for all supported devices. The entrypoint copies it into
# /usr/share/hexagon-dsp/conf.d as 00-arduino-dsp-binaries.yaml: the fastrpc parser keeps
# the last alphabetical match, so this file sorts first and acts as a fallback that any
# yaml provided by the host overrides.
COPY conf/hexagon-dsp-binaries.yaml /etc/fastrpc/hexagon-dsp-binaries.yaml

# Startup provisioning of the merged DSP view, and the entrypoint wrapper that runs it.
# Child images prepend the wrapper to their ENTRYPOINT (and so do the brick compose files
# when they override it).
COPY --chmod=755 scripts/provision-fastrpc-dsp.sh /provision-fastrpc-dsp.sh
COPY --chmod=755 scripts/aihub-onnx-entrypoint.sh /aihub-onnx-entrypoint.sh

# Pre-create the merged DSP view root, owned by arduino so images running as the non-root
# user can populate it at startup. The baked yaml's mode is pinned rather than inherited
# from the build context: the entrypoint copies it into a directory every uid has to be
# able to read, and `cp` starts from the source mode.
RUN set -ex; \
    chmod 644 /etc/fastrpc/hexagon-dsp-binaries.yaml; \
    mkdir -p /usr/share/hexagon-dsp/conf.d; \
    chown -R arduino:arduino /usr/share/hexagon-dsp

COPY --from=framework /opt/aihub-dist/requirements-base.txt .
COPY requirements.txt .

# requirements.txt is the complete, hash-locked package list for linux aarch64 / CPython
# 3.13 (see requirements.in): --no-deps --require-hashes installs exactly that and nothing
# else, so neither onnxruntime, onnxruntime-qnn nor a transitive dependency can drift under
# a rebuild, and the ~80 MB of sympy/mpmath/coloredlogs the wheels declare but never import
# stay out. OpenCV is in there too: the aihub framework needs cv2, so every runner on this
# base does, and one copy here beats one per runner.
RUN set -ex; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-base.txt; \
    /opt/venv/bin/pip install --no-cache-dir --no-deps --require-hashes -r requirements.txt; \
    /opt/venv/bin/python -m compileall -q /opt/venv/lib; \
    rm requirements-base.txt requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY --from=framework /opt/aihub-dist/aihub/ /opt/aihub/

# Prepare the app directory for child images
WORKDIR /app
COPY --from=framework /opt/aihub-dist/app/ ./

ENTRYPOINT [ "/aihub-onnx-entrypoint.sh", "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]
