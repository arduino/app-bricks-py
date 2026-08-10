# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Multi-stage Dockerfile for common QAIRT based services

FROM python:3.13-slim-trixie AS base

# Install common dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      curl libyaml-0-2 libbsd0 libmd0 libatomic1 strace netcat-openbsd && \
    apt-get clean; \
    apt-get dist-clean; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/lib/dpkg/info/*.md5sums \
        /var/lib/dpkg/*-old \
        /var/cache/apt/*.bin; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

# ============================
# Stage: Native Libraries (ARM64 only)
# ============================
FROM base AS native-libs

# QAIRT version to download and install
ENV QNP_VER=2.45.41.260507
ENV QNP_SDK_SHA256=8a4afa4e87d6653ba227fd0988f8ee51651e800669654a94d09cbc0a6fe9f9b2

# Tools for download/unzip
RUN apt-get update && \
    apt-get install -y wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DEPLOY_DIR="/mnt/work/deploy"
ENV ADSP_DIR=${DEPLOY_DIR}/usr/lib/rfsa/adsp
ENV DOWNLOAD_DIR="/mnt/work/downloads"
ENV QNP_SDK=v${QNP_VER}.zip

# Expected unzip layout (adjust if archive differs)
ENV QNN_ROOT=${DOWNLOAD_DIR}/qairt/${QNP_VER}
ENV QNN_LIB_PATH=${QNN_ROOT}/lib

# Ensure structure for downstream COPY
RUN mkdir -p ${DEPLOY_DIR}/usr/lib ${ADSP_DIR} /usr/include /usr/lib ${DOWNLOAD_DIR}

# Install and VALIDATE native libs
RUN set -eu && \
    echo "==> Clean previous artifacts"; \
    rm -rf "${DOWNLOAD_DIR}/qairt" || true; \
    rm -f  "${DOWNLOAD_DIR}/${QNP_SDK}" || true; \
    \
    echo "==> Download QAIRT ${QNP_VER}"; \
    wget -t 3 -T 300 -P "${DOWNLOAD_DIR}" \
      "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNP_VER}/${QNP_SDK}"; \
    \
    echo "==> Verify SDK checksum"; \
    echo "${QNP_SDK_SHA256}  ${DOWNLOAD_DIR}/${QNP_SDK}" | sha256sum -c -; \
    \
    echo "==> Unzip SDK"; \
    cd "${DOWNLOAD_DIR}" && unzip -q "${QNP_SDK}" && rm -f "${QNP_SDK}"; \
    \
    echo '==> Validate expected directory layout'; \
    test -d "${QNN_ROOT}" || { echo "ERROR: ${QNN_ROOT} missing"; ls -la "${DOWNLOAD_DIR}/qairt" || true; exit 1; }; \
    test -d "${QNN_LIB_PATH}" || { echo "ERROR: ${QNN_LIB_PATH} missing"; ls -la "${QNN_ROOT}" || true; exit 1; }; \
    \
    echo '==> Copy aarch64 QNN libs to deploy'; \
    cp -v "${QNN_LIB_PATH}/aarch64-oe-linux-gcc11.2/"* "${DEPLOY_DIR}/usr/lib/"; \
    \
    echo '==> Copy ADSP (Hexagon) unsigned binaries'; \
    for hv in hexagon-v73 hexagon-v75; do \
      if [ -d "${QNN_LIB_PATH}/${hv}/unsigned" ]; then \
        cp -v "${QNN_LIB_PATH}/${hv}/unsigned/"* "${ADSP_DIR}/"; \
      else \
        echo "WARN: ${hv}/unsigned not found for ${QNP_VER}"; \
      fi; \
    done; \
    \
    echo '==> Validate ADSP dir non-empty and contains .so'; \
    test -d "${ADSP_DIR}" && test "$(find "${ADSP_DIR}" -type f -name '*.so' | wc -l)" -ge 1 || { \
      echo "ERROR: No ADSP .so files copied into ${ADSP_DIR}"; exit 1; \
    }; \
    \
    echo "==> QAIRT ${QNP_VER} install and validation complete"

# Fail-fast: ADSP sanity (sprpc/skel presence optional)
RUN if ! find "${ADSP_DIR}" -type f -name "*.so" | grep -q . ; then \
      echo "ERROR: No .so files in ${ADSP_DIR}"; exit 1; \
    fi; \
    if ! (find "${ADSP_DIR}" -type f \( -name "*sprpc*.so" -o -name "*skel*.so" \) | grep -q .) ; then \
      echo "WARNING: No *sprpc*/*skel* libs detected in ${ADSP_DIR}. Continuing."; \
    fi

# ============================
# Stage: Build fastrpc (libcdsprpc.so + libadsprpc.so)
# ============================
# Builds fastrpc from source. Provides libcdsprpc.so and libadsprpc.so
# which are required by libtranslation.so at runtime.
# Inherits from base to reuse the already-pulled image and apt-get upgrade layer.
FROM base AS fastrpc-builder

# fastrpc version to build (git tag) as per QAIRT framework
ENV FASTRPC_VER=1.0.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git ca-certificates \
      autoconf automake libtool make gcc pkg-config \
      libyaml-dev libmd-dev libbsd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v${FASTRPC_VER} https://github.com/quic/fastrpc.git \
    && cd fastrpc \
    && autoreconf -is \
    && CFLAGS='-UMACHINE_NAME_PATH -DMACHINE_NAME_PATH=\"/run/device-model\"' \
       ./configure --prefix=/usr --libdir=/usr/lib \
    && make -j1 \
    && make install \
    && rm -rf /fastrpc

# ============================
# Stage: Final Runtime Image
# ============================
# Inherits from base to reuse the already-pulled image and apt-get upgrade layer.
# base already ran apt-get upgrade so we don't need to repeat it here.
FROM base AS runtime

# Create a non-root user for Arduino services
RUN groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino; \
    chown -R arduino:arduino /home/arduino

# Add native compiled libraries from the native-libs stage
COPY --from=native-libs /mnt/work/deploy/usr /usr

# Copy fastrpc libs built from source
COPY --from=fastrpc-builder /usr/lib/libcdsprpc* /usr/lib/
COPY --from=fastrpc-builder /usr/lib/libadsprpc* /usr/lib/

RUN ldconfig

# Bake fastrpc DSP path config for all supported devices.
# The entrypoint copies this to /usr/share/qcom/conf.d/ at runtime
# after the bind-mount is in place, so fastrpc_shell resolves from the host.
RUN mkdir -p /etc/fastrpc && \
    { \
      echo '# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies'; \
      echo '#'; \
      echo '# SPDX-License-Identifier: MPL-2.0'; \
      echo 'machines:'; \
      echo '  Qualcomm Technologies, Inc. Monaco Monza addons:'; \
      echo '    DSP_LIBRARY_PATH: qcs8300/Qualcomm/QCS8300-RIDE/dsp'; \
      echo '  Qualcomm Technologies, Inc. Addons IQ 9075 EVK:'; \
      echo '    DSP_LIBRARY_PATH: sa8775p/Qualcomm/SA8775P-RIDE/dsp'; \
      echo '  Qualcomm Technologies, Inc. IQ8 8275 Pro SKU EVK:'; \
      echo '    DSP_LIBRARY_PATH: qcs8300/Qualcomm/QCS8300-RIDE/dsp'; \
    } > /etc/fastrpc/hexagon-dsp-binaries.yaml

# Use container libraries
ENV ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
ENV CDSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
