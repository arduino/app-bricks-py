# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim AS builder

COPY ./requirements-base.txt /app/requirements.txt

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
    apt-get install build-essential libasound2-dev libalsaplayer-dev libzbar0 binutils portaudio19-dev -y; \
    pip install --no-cache-dir -r /app/requirements.txt && rm /app/requirements.txt && pip cache purge; \
    # remove debug symbols from shared objects and reduce size
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so" -exec strip -s {} +; \
    find /usr/local/lib/python3.13/site-packages -type f -name "*.so.*" -exec strip -s {} +; \
    # remove unneeded files/packages
    apt remove build-essential libasound2-dev libalsaplayer-dev binutils -y; \
    apt-get clean; \
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +

FROM python:3.13-slim AS production

# Copy the installed dependencies from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN set -ex; \
    groupadd -g 44 video || true; \
    groupadd -g 29 audio || true; \
    groupadd -g 20 dialout || true; \
    groupadd -g 991 render || true; \
    groupadd -g 1000 arduino || true; \
    groupadd -g 1001 gpiod || true; \
    useradd -u 1000 -g arduino -G audio,video,dialout,render,gpiod -ms /bin/bash arduino; \
    apt-get update -y; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y libglib2.0-0 alsa-utils inotify-tools wget libzbar0 portaudio19-dev; \
    pip cache purge; \
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
