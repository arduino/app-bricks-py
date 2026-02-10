# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

FROM debian:trixie AS production

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
    apt install --no-install-recommends -y curl libgomp1 ca-certificates; \
    update-ca-certificates; \    
    apt-get autoclean -y; \
    apt-get autoremove -y; \
    apt-get dist-clean; \
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
        /var/cache/apt/*.bin
