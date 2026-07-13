# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Connect to an authenticated remote MCP server"
# EXAMPLE_REQUIRES = "Requires an access token/API key for the target MCP server."

import os

from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_utils import Logger, App


logger = Logger(name="authenticated_mcp_client_example")

# Bearer token auth (e.g. a GitHub PAT or a Stripe restricted key).
# `token` is added as an "Authorization: Bearer <token>" header.
github = HTTPEndpoint(
    name="github",
    url="https://api.githubcopilot.com/mcp/",
    token=os.getenv("GITHUB_MCP_PAT"),
)

# Custom-header auth: some providers use their own header scheme instead of a bearer token.
# datadog = HTTPEndpoint(
#     name="datadog",
#     url="https://<your-datadog-mcp-domain>/mcp",
#     headers={
#         "DD_API_KEY": os.getenv("DD_API_KEY", ""),
#         "DD_APPLICATION_KEY": os.getenv("DD_APP_KEY", ""),
#     },
# )

client = MCPClient(endpoints=[github])

logger.info(client.get_tools())

App.run()
