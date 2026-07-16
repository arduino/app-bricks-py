# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Use MCP tools with an LLM"
# EXAMPLE_REQUIRES = "Requires a valid API key to a cloud LLM service."

from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

# The MCP server is built and deployed with the app (see bricks/mcp-server);
# The model can call its tools while chatting: try "What time is it in Rome?".
mcp = MCPClient(endpoints=[HTTPEndpoint(name="clock", url="http://mcp-server:8080/mcp")])

llm = CloudLLM(
    model="google:gemini-2.5-flash",
    api_key="YOUR_API_KEY",  # Replace with your actual API key
    tools=mcp.get_tools(),
)


def ask_prompt():
    prompt = input("Enter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        raise StopIteration()
    print(llm.chat(prompt))
    print()


App.run(ask_prompt)
