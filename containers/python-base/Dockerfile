# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim AS builder

COPY ./requirements-base.txt /app/requirements.txt

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    # Install build dependencies
    apt-get install -y --no-install-recommends --no-install-suggests git pkg-config build-essential ca-certificates binutils; \
    # Install dependencies for building ALSA
    apt-get install -y --no-install-recommends --no-install-suggests libasound2-dev libalsaplayer-dev portaudio19-dev libzbar0; \
    # Install dependencies for building OpenCV with Gstreamer backend
    apt-get install -y --no-install-recommends --no-install-suggests libavcodec-dev libavformat-dev libavutil-dev libswscale-dev; \
    apt-get install -y --no-install-recommends --no-install-suggests libjpeg62-turbo-dev libpng-dev libtiff-dev libwebp-dev libavif-dev libopenblas-dev liblapacke-dev; \
    apt-get install -y --no-install-recommends --no-install-suggests libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev; \
    # Build OpenCV (pinned at 4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8)
    git clone --depth 1 https://github.com/opencv/opencv-python.git; \
    cd opencv-python; \
    git fetch --depth 1 origin 4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8; \
    git checkout 4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8; \
    git submodule update --init --recursive --depth 1; \
    cd ..; \
    export OPENCV_VERSION=4.13.0; \
    export ENABLE_CONTRIB=0; \
    export ENABLE_HEADLESS=1; \
    export CMAKE_ARGS="-DWITH_GSTREAMER=ON -DWITH_FFMPEG=ON -DWITH_CAROTENE=ON -DWITH_LAPACK=ON -DBUILD_opencv_python3=ON -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_CUDA=OFF -DBUILD_opencv_java=OFF -DWITH_VTK=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnn_modern=OFF"; \
    pip wheel opencv-python --verbose --no-deps; \
    mv opencv_python_headless*.whl opencv_python_headless.whl; \
    # Install Python dependencies
    pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt; \
    # Strip debug symbols, tests and remove unneeded packages to reduce size
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so.*" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} + \
    apt remove -y git pkg-config build-essential ca-certificates binutils libasound2-dev libalsaplayer-dev portaudio19-dev libjpeg62-turbo-dev libpng-dev libtiff-dev libwebp-dev libavif-dev libopenblas-dev liblapacke-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev; \
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
    # Install dependencies for running ALSA
    apt-get install -y --no-install-recommends --no-install-suggests alsa-utils libportaudiocpp0; \
    # Install dependencies for running OpenCV with Gstreamer backend
    apt-get install -y --no-install-recommends --no-install-suggests libavcodec61 libavformat61 libavutil59 libswscale8; \
    apt-get install -y --no-install-recommends --no-install-suggests libjpeg62-turbo libtiff6 libpng16-16t64 libwebp7 libwebpdecoder3 libwebpdemux2 libwebpmux3 libavif16 libopenblas0 liblapacke; \
    apt-get install -y --no-install-recommends --no-install-suggests libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 gstreamer1.0-plugins-good gstreamer1.0-libav gstreamer1.0-libcamera libcamera0.4 libcamera-ipa libcamera-tools; \
    # Install other runtime dependencies
    apt-get install -y --no-install-recommends --no-install-suggests libglib2.0-0 inotify-tools wget libzbar0; \
    apt-get dist-clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    mkdir -p /home/app/.fonts; \
    wget -O /home/app/.fonts/OpenSans.ttf https://raw.githubusercontent.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf; \
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
