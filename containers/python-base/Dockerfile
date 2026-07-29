# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim AS builder

ARG OPENCV_WHL_URL
ARG OPENCV_WHL_DIGEST

COPY ./requirements-base.txt /app/requirements.txt

ADD --checksum=${OPENCV_WHL_DIGEST} ${OPENCV_WHL_URL} /tmp/opencv-wheel/

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install build dependencies
    apt-get install -y --no-install-recommends --no-install-suggests build-essential ca-certificates binutils; \
    # Install dependencies for building ALSA
    apt-get install -y --no-install-recommends --no-install-suggests libasound2-dev libalsaplayer-dev portaudio19-dev libzbar0; \
    # Install build dependencies for building PyGObject
    apt-get install -y --no-install-recommends --no-install-suggests pkg-config libgirepository-2.0-dev libglib2.0-dev; \
    # Install Python dependencies (requirements file + prefetched OpenCV wheel)
    pip install --no-cache-dir uv; \
    uv pip install --system --no-cache-dir -r /app/requirements.txt /tmp/opencv-wheel/*.whl; \
    printf 'opencv-python-headless==%s\n' "$(python -c 'from importlib.metadata import version; print(version("opencv-python-headless"))')" > /etc/uv-constraints.txt; \
    rm -rf /app/requirements.txt /tmp/opencv-wheel; \
    # Build PyGObject without cairo support
    uv pip install --system --no-cache-dir meson-python ninja; \
    uv pip install --system --no-cache-dir --no-build-isolation --no-deps -C setup-args="-Dpycairo=disabled" PyGObject; \
    uv pip uninstall --system meson-python meson ninja pyproject-metadata packaging; \
    # Strip debug symbols, tests and remove unneeded packages to reduce size
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so.*" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    apt remove -y build-essential ca-certificates binutils libasound2-dev libalsaplayer-dev portaudio19-dev pkg-config libgirepository-2.0-dev libglib2.0-dev; \
    apt-get clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y


FROM python:3.13-slim AS production

ARG LIBCAMERA_DEB_URL
ARG LIBCAMERA_DEB_DIGEST
ARG LIBCAMERA_IPA_DEB_URL
ARG LIBCAMERA_IPA_DEB_DIGEST
ARG GSTREAMER_LIBCAMERA_DEB_URL
ARG GSTREAMER_LIBCAMERA_DEB_DIGEST
ARG GSTREAMER_QTIQMMFSRC_DEB_URL
ARG GSTREAMER_QTIQMMFSRC_DEB_DIGEST
ARG GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_URL
ARG GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_DIGEST

ADD --checksum=${LIBCAMERA_DEB_DIGEST} ${LIBCAMERA_DEB_URL} /tmp/libcamera0.7.deb
ADD --checksum=${LIBCAMERA_IPA_DEB_DIGEST} ${LIBCAMERA_IPA_DEB_URL} /tmp/libcamera-ipa.deb
ADD --checksum=${GSTREAMER_LIBCAMERA_DEB_DIGEST} ${GSTREAMER_LIBCAMERA_DEB_URL} /tmp/gstreamer1.0-libcamera.deb
ADD --checksum=${GSTREAMER_QTIQMMFSRC_DEB_DIGEST} ${GSTREAMER_QTIQMMFSRC_DEB_URL} /tmp/qtiqmmfsrc.deb
ADD --checksum=${GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_DIGEST} ${GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_URL} /tmp/libgstreamer-plugins-base.deb
ADD https://raw.githubusercontent.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf /home/app/.fonts/OpenSans.ttf

# Copy the installed dependencies from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /etc/uv-constraints.txt /etc/uv-constraints.txt

ENV UV_CONSTRAINT=/etc/uv-constraints.txt
ENV PIP_CONSTRAINT=/etc/uv-constraints.txt

RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install dependencies for running ALSA
    apt-get install -y --no-install-recommends --no-install-suggests alsa-utils libportaudiocpp0; \
    # Install dependencies for pipewire
    apt-get install -y --no-install-recommends --no-install-suggests pipewire-alsa; \
    # Install dependencies for running OpenCV with Gstreamer backend
    apt-get install -y --no-install-recommends --no-install-suggests libavcodec61 libavformat61 libavutil59 libswscale8; \
    apt-get install -y --no-install-recommends --no-install-suggests libjpeg62-turbo libtiff6 libpng16-16t64 libwebp7 libwebpdecoder3 libwebpdemux2 libwebpmux3 libopenblas0 liblapacke; \
    apt-get install -y --no-install-recommends --no-install-suggests libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 gstreamer1.0-plugins-good gstreamer1.0-plugins-base-apps; \
    # Install dependencies for running CSI cameras (libcamera, gstreamer introspection, qtiqmmfsrc)
    apt-get install -y --no-install-recommends --no-install-suggests libgirepository-2.0-0 gir1.2-glib-2.0 gir1.2-gstreamer-1.0 libpisp-common libboost-filesystem1.83.0 libboost-log1.83.0 libboost-thread1.83.0 libpisp1 libegl1 libgles2 libyaml-0-2 liblttng-ust1 libdw1t64 libyuv0; \
    dpkg -i /tmp/libcamera0.7.deb /tmp/libcamera-ipa.deb /tmp/gstreamer1.0-libcamera.deb; \
    apt-get install -y --no-install-recommends --no-install-suggests --allow-downgrades /tmp/qtiqmmfsrc.deb /tmp/libgstreamer-plugins-base.deb; \
    rm /tmp/libcamera0.7.deb /tmp/libcamera-ipa.deb /tmp/gstreamer1.0-libcamera.deb /tmp/qtiqmmfsrc.deb /tmp/libgstreamer-plugins-base.deb; \
    # Install other runtime dependencies
    apt-get install -y --no-install-recommends --no-install-suggests libglib2.0-0 inotify-tools libzbar0; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    dpkg -r --force-depends libegl-mesa0 mesa-libgallium libllvm19; \
    apt-get dist-clean; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/cache/debconf/* \
        /var/lib/dpkg/info/* \
        /var/cache/apt/*.bin; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib; \
    # HACK: Remove VPX plugin
    rm -f /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstvpx.so; \
    # Ensure libvpx and libx265 are not present
    dpkg -r --force-depends libx265-215 libvpx9

COPY libcamera.yaml /etc/libcamera/configuration.yaml
