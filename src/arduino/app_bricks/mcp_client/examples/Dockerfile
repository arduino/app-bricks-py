# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM node:20-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir mcp-proxy

RUN npm install -g @modelcontextprotocol/server-filesystem

WORKDIR /app
RUN mkdir -p /data

EXPOSE 8080

CMD ["mcp-proxy", "--host", "0.0.0.0", "--transport", "streamablehttp", "--port", "8080", "mcp-server-filesystem", "/data"]
