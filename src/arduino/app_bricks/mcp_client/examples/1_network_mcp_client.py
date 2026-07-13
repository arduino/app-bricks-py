# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "List tools from a remote MCP server over HTTP"

from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_utils import Logger, App

logger = Logger(name="network_mcp_client_example")

external_mcp = HTTPEndpoint(name="filesystem_proxy", url="http://localhost:8080/mcp")

client = MCPClient(endpoints=[external_mcp])

logger.info(client.get_tools())

App.run()
