# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0


FROM python:3.13-slim AS builder

ARG OPENCV_GITHUB_REPO=arduino/app-bricks-py
ARG OPENCV_RELEASE_VERSION=4.13.0.92
ARG OPENCV_PYTHON_VERSION=cp313
ARG OPENCV_WHL_SHA256=0000000000000000000000000000000000000000000000000000000000000000

COPY ./requirements-base.txt /app/requirements.txt

ADD --checksum=sha256:${OPENCV_WHL_SHA256} \
    https://github.com/${OPENCV_GITHUB_REPO}/releases/download/opencv/${OPENCV_RELEASE_VERSION}/opencv_python_headless-${OPENCV_RELEASE_VERSION}-${OPENCV_PYTHON_VERSION}-${OPENCV_PYTHON_VERSION}-linux_aarch64.whl \
    /tmp/opencv_python_headless.whl

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install build dependencies
    apt-get install -y --no-install-recommends --no-install-suggests build-essential ca-certificates binutils; \
    # Install dependencies for building ALSA
    apt-get install -y --no-install-recommends --no-install-suggests libasound2-dev libalsaplayer-dev portaudio19-dev libzbar0; \
    # Install Python dependencies (requirements file + prefetched OpenCV wheel)
    pip install --no-cache-dir uv; \
    uv pip install --system --no-cache-dir -r /app/requirements.txt /tmp/opencv_python_headless.whl; \
    rm /app/requirements.txt /tmp/opencv_python_headless.whl; \
    # Strip debug symbols, tests and remove unneeded packages to reduce size
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so.*" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    apt remove -y build-essential ca-certificates binutils libasound2-dev libalsaplayer-dev portaudio19-dev; \
    apt-get clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y


FROM python:3.13-slim AS production

ARG LIBCAMERA_GITHUB_REPO=arduino/app-bricks-py
ARG LIBCAMERA_RELEASE_VERSION=0.7.0-4

ADD https://github.com/${LIBCAMERA_GITHUB_REPO}/releases/download/libcamera%2F${LIBCAMERA_RELEASE_VERSION}/libcamera0.7_${LIBCAMERA_RELEASE_VERSION}_arm64.deb /tmp/libcamera0.7.deb
ADD https://github.com/${LIBCAMERA_GITHUB_REPO}/releases/download/libcamera%2F${LIBCAMERA_RELEASE_VERSION}/libcamera-ipa_${LIBCAMERA_RELEASE_VERSION}_arm64.deb /tmp/libcamera-ipa.deb
ADD https://github.com/${LIBCAMERA_GITHUB_REPO}/releases/download/libcamera%2F${LIBCAMERA_RELEASE_VERSION}/gstreamer1.0-libcamera_${LIBCAMERA_RELEASE_VERSION}_arm64.deb /tmp/gstreamer1.0-libcamera.deb
ADD https://raw.githubusercontent.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf /home/app/.fonts/OpenSans.ttf

# Copy the installed dependencies from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install dependencies for running ALSA
    apt-get install -y --no-install-recommends --no-install-suggests alsa-utils libportaudiocpp0; \
    # Install dependencies for running OpenCV with Gstreamer backend
    apt-get install -y --no-install-recommends --no-install-suggests libavcodec61 libavformat61 libavutil59 libswscale8; \
    apt-get install -y --no-install-recommends --no-install-suggests libjpeg62-turbo libtiff6 libpng16-16t64 libwebp7 libwebpdecoder3 libwebpdemux2 libwebpmux3 libopenblas0 liblapacke; \
    apt-get install -y --no-install-recommends --no-install-suggests libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 gstreamer1.0-plugins-good gstreamer1.0-plugins-base-apps; \
    # Install dependencies for running libcamera
    apt-get install -y libpisp-common libboost-filesystem1.83.0 libboost-log1.83.0 libboost-thread1.83.0 libpisp1 libegl1 libgles2 libyaml-0-2 liblttng-ust1 libdw1t64 libyuv0; \
    dpkg -i /tmp/libcamera0.7.deb /tmp/libcamera-ipa.deb /tmp/gstreamer1.0-libcamera.deb; \
    rm /tmp/libcamera0.7.deb /tmp/libcamera-ipa.deb /tmp/gstreamer1.0-libcamera.deb; \
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
    python -m compileall /usr/local/lib

COPY libcamera.yaml /etc/libcamera/configuration.yaml
