# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Basic mcp client usage example"

from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_utils import App

external_mcp = HTTPEndpoint(name="filesystem_proxy", url="http://localhost:8080/mcp")

client = MCPClient(clients=[external_mcp])

print(client.list_tools())

App.run()
