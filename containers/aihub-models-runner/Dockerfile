# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

FROM ubuntu:24.04

COPY requirements-base.txt .

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests netcat-openbsd python3 python3-pip  python3-venv python-is-python3; \
    # Install Qualcomm AI Engine dependencies
    apt-get install -y --no-install-recommends --no-install-suggests software-properties-common \
    && apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa \
    && apt-get update; \
    apt-get install -y --no-install-recommends --no-install-suggests libqnn1 qcom-fastrpc1; \
    # Install Python dependencies
    python -m pip install --no-cache-dir --break-system-packages -r requirements-base.txt \
    && python -m compileall /usr/local/bin \
    && python -m compileall /usr/local/lib \
    && rm requirements-base.txt; \
    # Clean up
    apt-get remove -y software-properties-common python3-blinker \
    # Remove packages that are no longer needed
    && apt-get autoremove -y \
    && apt-get dist-clean \
    && rm -rf /var/lib/apt/lists/*; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv

# Prepare the user
# RUN groupadd -g 1000 arduino && useradd -u 1000 -g arduino -ms /bin/bash arduino
# USER arduino

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY main.py inference.py requirements.txt ./

ENTRYPOINT [ "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]