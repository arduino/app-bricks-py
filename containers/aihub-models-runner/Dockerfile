# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

FROM ubuntu:24.04

# Install system dependencies
RUN apt update \
    && apt install -y --no-install-recommends --no-install-suggests software-properties-common \
    && apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa \
    && apt update \
    && apt install -y --no-install-recommends --no-install-suggests \
    netcat-openbsd \
    python3 \
    python3-pip \
    python3-venv \
    libqnn1 \
    qcom-fastrpc1
    
# Install Python dependencies and precompile to speed up startup time
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m compileall /usr/local/bin \
    && python -m compileall /usr/local/lib

# Clean up
RUN apt remove -y \
    software-properties-common \
    && apt autoremove -y \
    && apt dist-clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 arduino \
    && useradd -u 1000 -g arduino -ms /bin/bash arduino

ENV PYTHONPATH="/opt/framework"
COPY framework/ /opt/framework/

USER arduino

CMD ["python", "main.py", "--verbose", "--input", "websocket", "--output", "mjpeg", "websocket"]