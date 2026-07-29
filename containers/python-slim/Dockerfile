# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM python:3.13-slim AS production

RUN set -ex; \
    apt-get update -y; \
    apt-get upgrade -y; \
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
        /var/cache/debconf/* \
        /var/cache/apt/*.bin; \
    # keep the rest of /var/lib/dpkg/info, or derived images cannot apt-get upgrade
    rm -fr /var/lib/dpkg/info/*.symbols \
        /var/lib/dpkg/info/*.shlibs \
        /var/lib/dpkg/info/*.templates; \
    find /usr/share/doc -type f ! -name 'copyright' -delete; \
    find /usr/share/doc -type d -empty -delete; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib
