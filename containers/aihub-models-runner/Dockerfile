# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG DEBIAN_FRONTEND=noninteractive

FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests software-properties-common \
    && apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
    netcat-openbsd \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    libqnn1 \
    qcom-fastrpc1
    
# Install Python dependencies and precompile to speed up startup time
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m compileall /usr/local/bin \
    && python -m compileall /usr/local/lib

# Clean up
RUN apt-get remove -y \
    software-properties-common \
    && apt-get autoremove -y \
    && apt-get dist-clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 arduino \
    && useradd -u 1000 -g arduino -ms /bin/bash arduino

ENV PYTHONPATH="/opt/aihub"
COPY aihub/ /opt/aihub/

USER arduino

CMD ["python", "main.py", "--verbose", "--input", "websocket", "--output", "mjpeg", "websocket"]