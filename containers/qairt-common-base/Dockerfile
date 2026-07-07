# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Multi-stage Dockerfile for common QAIRT based services

FROM python:3.13-slim-trixie AS base

# Install common dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Stage: Native Libraries (ARM64 only)
# ============================
FROM base AS native-libs

# Inherit the global default or --build-arg
ARG INSTALL_NATIVE_LIBS=true

# Tools for download/unzip
RUN if [ "$INSTALL_NATIVE_LIBS" = "true" ]; then \
    apt-get update && \
    apt-get install -y wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*; \
    fi

ENV VLM_BUILD_DIR="/mnt/work/build"
ENV VLM_DEPLOY_DIR="/mnt/work/deploy"
ENV VLM_ADSP_DIR=${VLM_DEPLOY_DIR}/usr/lib/rfsa/adsp
ENV VLM_QNP_VER=2.46.0.260424
ENV VLM_DOWNLOAD_DIR="/mnt/work/downloads"
ENV VLM_QNP_SDK=v${VLM_QNP_VER}.zip
# Expected unzip layout (adjust if archive differs)
ENV VLM_QNN_ROOT=${VLM_DOWNLOAD_DIR}/qairt/${VLM_QNP_VER}
ENV VLM_QNN_LIB_PATH=${VLM_QNN_ROOT}/lib

# Show variables
RUN echo "INSTALL_NATIVE_LIBS=${INSTALL_NATIVE_LIBS}" && \
    echo "VLM_QNP_VER=${VLM_QNP_VER}" && \
    echo "VLM_QNP_SDK=${VLM_QNP_SDK}" && \
    echo "VLM_QNN_ROOT=${VLM_QNN_ROOT}" && \
    echo "VLM_QNN_LIB_PATH=${VLM_QNN_LIB_PATH}"

# Ensure structure for downstream COPY
RUN mkdir -p ${VLM_DEPLOY_DIR}/usr/lib ${VLM_ADSP_DIR} /usr/include /usr/lib ${VLM_DOWNLOAD_DIR}

# Install and VALIDATE native libs
RUN set -eu && \
    echo "INSTALL_NATIVE_LIBS=${INSTALL_NATIVE_LIBS}" && \
    if [ "${INSTALL_NATIVE_LIBS}" = "true" ]; then \
      echo "==> Clean previous artifacts"; \
      rm -rf "${VLM_DOWNLOAD_DIR}/qairt" || true; \
      rm -f  "${VLM_DOWNLOAD_DIR}/${VLM_QNP_SDK}" || true; \
      \
      echo "==> Download QAIRT ${VLM_QNP_VER}"; \
      wget -t 3 -T 300 -P "${VLM_DOWNLOAD_DIR}" \
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${VLM_QNP_VER}/${VLM_QNP_SDK}"; \
      \
      echo "==> Unzip SDK"; \
      cd "${VLM_DOWNLOAD_DIR}" && unzip -q "${VLM_QNP_SDK}" && rm -f "${VLM_QNP_SDK}"; \
      \
      echo '==> Validate expected directory layout'; \
      test -d "${VLM_QNN_ROOT}" || { echo "ERROR: ${VLM_QNN_ROOT} missing"; ls -la "${VLM_DOWNLOAD_DIR}/qairt" || true; exit 1; }; \
      test -d "${VLM_QNN_LIB_PATH}" || { echo "ERROR: ${VLM_QNN_LIB_PATH} missing"; ls -la "${VLM_QNN_ROOT}" || true; exit 1; }; \
      \
      echo '==> Copy aarch64 QNN libs to deploy'; \
      cp -v "${VLM_QNN_LIB_PATH}/aarch64-oe-linux-gcc11.2/"* "${VLM_DEPLOY_DIR}/usr/lib/"; \
      \
      echo '==> Copy ADSP (Hexagon) unsigned binaries'; \
      for hv in hexagon-v73 hexagon-v75; do \
        if [ -d "${VLM_QNN_LIB_PATH}/${hv}/unsigned" ]; then \
          cp -v "${VLM_QNN_LIB_PATH}/${hv}/unsigned/"* "${VLM_ADSP_DIR}/"; \
        else \
          echo "WARN: ${hv}/unsigned not found for ${VLM_QNP_VER}"; \
        fi; \
      done; \
      \
      echo '==> Validate ADSP dir non-empty and contains .so'; \
      test -d "${VLM_ADSP_DIR}" && test "$(find "${VLM_ADSP_DIR}" -type f -name '*.so' | wc -l)" -ge 1 || { \
        echo "ERROR: No ADSP .so files copied into ${VLM_ADSP_DIR}"; exit 1; \
      }; \
      \
      echo "==> QAIRT ${VLM_QNP_VER} install and validation complete"; \
    else \
      echo "Skipping native libs install (INSTALL_NATIVE_LIBS!=true)"; \
    fi

# Fail-fast: ADSP sanity (sprpc/skel presence optional)
RUN if [ "$INSTALL_NATIVE_LIBS" = "true" ]; then \
      if ! find "${VLM_ADSP_DIR}" -type f -name "*.so" | grep -q . ; then \
        echo "ERROR: No .so files in ${VLM_ADSP_DIR}"; exit 1; \
      fi; \
      if ! (find "${VLM_ADSP_DIR}" -type f \( -name "*sprpc*.so" -o -name "*skel*.so" \) | grep -q .) ; then \
        echo "WARNING: No *sprpc*/*skel* libs detected in ${VLM_ADSP_DIR}. Continuing."; \
      fi; \
    fi

# ============================
# Stage: Build fastrpc (libcdsprpc.so + libadsprpc.so)
# ============================
# Builds fastrpc from source. Provides libcdsprpc.so and libadsprpc.so
# which are required by libtranslation.so at runtime.
# Inherits from base to reuse the already-pulled ubuntu:24.04 image and apt-get upgrade layer.
FROM base AS fastrpc-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git ca-certificates \
      autoconf automake libtool make gcc pkg-config \
      libyaml-dev libmd-dev libbsd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/quic/fastrpc.git \
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
# Inherits from base to reuse the already-pulled ubuntu:24.04 image and apt-get upgrade layer.
# base already ran apt-get upgrade so we don't need to repeat it here.
FROM base AS runtime

ARG INSTALL_NATIVE_LIBS=true

# Single combined layer: runtime deps only.
# base already ran apt-get upgrade. No build tools, no nodesource.
# Node binary is copied from api-builder.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl libyaml-0-2 libbsd0 libmd0 libatomic1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash -u 1000 iot-user

COPY --from=native-libs /mnt/work/deploy/usr /usr

# Copy fastrpc libs built from source
COPY --from=fastrpc-builder /usr/lib/libcdsprpc* /usr/lib/
COPY --from=fastrpc-builder /usr/lib/libadsprpc* /usr/lib/

# Bake fastrpc DSP path config for all supported devices.
# The entrypoint copies this to /usr/share/qcom/conf.d/ at runtime
# after the bind-mount is in place, so fastrpc_shell resolves from the host.
RUN mkdir -p /etc/fastrpc && \
    { \
      echo '# SPDX-License-Identifier: BSD-3-Clause-Clear'; \
      echo '# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.'; \
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
