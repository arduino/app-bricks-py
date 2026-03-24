# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim AS builder

ARG OPENCV_REPO_OWNER=arduino
ARG OPENCV_VERSION=4.13.0.92
ARG PYTHON_VERSION=cp313

COPY ./requirements-base.txt /app/requirements.txt

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install build dependencies
    apt-get install -y --no-install-recommends --no-install-suggests build-essential ca-certificates binutils; \
    # Install dependencies for building ALSA
    apt-get install -y --no-install-recommends --no-install-suggests libasound2-dev libalsaplayer-dev portaudio19-dev libzbar0; \
    # Replace placeholder with actual OpenCV wheel URL (full wheel name with platform details)
    sed -i "s|OPENCV_WHEEL_URL_PLACEHOLDER|https://github.com/${OPENCV_REPO_OWNER}/app-bricks-py/releases/download/opencv/${OPENCV_VERSION}/opencv_python_headless-${OPENCV_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_aarch64.whl|g" /app/requirements.txt; \
    # Install Python dependencies
    pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    # Strip debug symbols, tests and remove unneeded packages to reduce size
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so.*" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    apt remove -y build-essential ca-certificates binutils libasound2-dev libalsaplayer-dev portaudio19-dev; \
    apt-get clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y

FROM python:3.13-slim AS production

# Copy the installed dependencies from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino; \
    apt-get update -y; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends --no-install-suggests wget; \
    # Install dependencies for running ALSA
    apt-get install -y --no-install-recommends --no-install-suggests alsa-utils libportaudiocpp0; \
    # Install dependencies for running OpenCV with Gstreamer backend
    apt-get install -y --no-install-recommends --no-install-suggests libavcodec61 libavformat61 libavutil59 libswscale8; \
    apt-get install -y --no-install-recommends --no-install-suggests libjpeg62-turbo libtiff6 libpng16-16t64 libwebp7 libwebpdecoder3 libwebpdemux2 libwebpmux3 libavif16 libopenblas0 liblapacke; \
    apt-get install -y --no-install-recommends --no-install-suggests libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 gstreamer1.0-plugins-good gstreamer1.0-plugins-base-apps gstreamer1.0-libav; \
    # Install dependencies for running libcamera
    apt-get install -y libpisp-common libboost-filesystem1.83.0 libboost-log1.83.0 libboost-thread1.83.0 libpisp1 libegl1 libgles2 libyaml-0-2 liblttng-ust1 libdw1t64; \
    wget -O libcamera0.7_0.7.0-3_arm64.deb https://github.com/robgee86/app-bricks-py/releases/download/libcamera%2F0.7/libcamera0.7_0.7.0-3_arm64.deb; \
    wget -O libcamera-ipa_0.7.0-3_arm64.deb https://github.com/robgee86/app-bricks-py/releases/download/libcamera%2F0.7/libcamera-ipa_0.7.0-3_arm64.deb; \
    wget -O gstreamer1.0-libcamera_0.7.0-3_arm64.deb https://github.com/robgee86/app-bricks-py/releases/download/libcamera%2F0.7/gstreamer1.0-libcamera_0.7.0-3_arm64.deb; \
    dpkg -i libcamera0.7_0.7.0-3_arm64.deb libcamera-ipa_0.7.0-3_arm64.deb gstreamer1.0-libcamera_0.7.0-3_arm64.deb; \
    # Install other runtime dependencies
    apt-get install -y --no-install-recommends --no-install-suggests libglib2.0-0 inotify-tools libzbar0; \
    mkdir -p /home/app/.fonts; \
    wget -O /home/app/.fonts/OpenSans.ttf https://raw.githubusercontent.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf; \
    apt-get remove -y wget; \
    apt-get dist-clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    # cleanup unnecessary files to reduce image size
    rm -rf /tmp/* /var/tmp/*; \
    rm -rf /var/lib/apt/lists/*; \
    rm -fr /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb \
        /usr/share/doc/* \
        /usr/share/man/* \
        /usr/share/locale/* \
        /var/cache/debconf/* \
        /var/lib/dpkg/info/* \      
        /var/cache/apt/*.bin; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib
