# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

COPY ./run.sh /run.sh
COPY ./requirements-base.txt /app/requirements.txt
COPY ./arduino*.whl /app/
# Copy the Streamlit configuration file with customized settings in home app shared directory
COPY ./streamlit/config.toml /home/app/.streamlit/config.toml

RUN pip install --no-cache-dir -r /app/requirements.txt; \
    bash -c 'WHL_FILES=(/app/*.whl) && pip install --no-cache-dir "${WHL_FILES[@]}"[all] && pip cache purge'; \
    rm /app/requirements.txt; \
    rm /app/*.whl; \
    # fix permissions for /home/app
    chown arduino:arduino -R /home/app; \
    chmod 777 -R /home/app; \
    # ensure permissions for /run.sh
    chmod 755 /run.sh; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

USER arduino

WORKDIR /app

ENV HOME=/home/app
ENV USER=arduino

ENTRYPOINT [ "/run.sh" ]
