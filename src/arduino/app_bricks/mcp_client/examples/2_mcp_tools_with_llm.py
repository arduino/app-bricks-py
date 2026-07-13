# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Use MCP tools with an LLM"
# EXAMPLE_REQUIRES = "Requires a reachable MCP server and a valid API key to a cloud LLM service."

from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

# Expose the tools of a remote MCP server as LangChain tools.
# See the Dockerfile in this folder for a ready-to-run filesystem MCP server on port 8080.
mcp = MCPClient(clients=[HTTPEndpoint(name="filesystem", url="http://localhost:8080/mcp")])

# Hand the discovered MCP tools to the LLM: the model can now call them while chatting.
# The same `tools=mcp.get_tools()` also works with the on-device `LargeLanguageModel` brick.
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
