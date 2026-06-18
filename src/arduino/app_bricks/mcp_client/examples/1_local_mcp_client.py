# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "List tools from a local MCP server"

from arduino.app_bricks.mcp_client import MCPClient, LocalPythonMCPEndpoint
from arduino.app_utils import Logger, App


logger = Logger(name="local_mcp_client_example")

local = LocalPythonMCPEndpoint(name="local_math_server", script_path="math_server.py")

client = MCPClient(clients=[local])

logger.info(client.get_tools())

App.run()
