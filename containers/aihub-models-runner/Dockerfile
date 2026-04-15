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
    && apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests libqnn1 qcom-fastrpc1; \
    # Clean up
    apt-get remove -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get dist-clean \
    && rm -rf /var/lib/apt/lists/*; \
    # Make the venv the default for child images
    python -m venv --system-site-packages /opt/venv; \
    # Install Python dependencies into the venv
    /opt/venv/bin/pip install --no-cache-dir -r requirements-base.txt; \
    /opt/venv/bin/python -m compileall /opt/venv/lib; \
    rm requirements-base.txt; \
    # Remove unneeded linux firmware
    dpkg -r --force-depends linux-firmware-dragonwing qcom-video-firmware tc9564-firmware firmware-qcm6490-msl firmware-qcom-hlosfw linux-firmware 

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV PYTHONPATH="/opt"
COPY aihub/ /opt/aihub/

# Prepare the app directory for child images
RUN mkdir -p /app
WORKDIR /app
COPY main.py inference.py requirements.txt ./

ENTRYPOINT [ "python", "main.py", "--input", "websocket", "--output", "mjpeg", "websocket" ]