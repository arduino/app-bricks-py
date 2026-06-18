# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Use MCP tools with an LLM"
# EXAMPLE_REQUIRES = "Requires a valid API key to a cloud LLM service."

from arduino.app_bricks.mcp_client import MCPClient, LocalPythonMCPEndpoint
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

# Expose the tools of a local MCP server (see math_server.py) as LangChain tools.
local = LocalPythonMCPEndpoint(name="math", script_path="math_server.py")
mcp = MCPClient(clients=[local])

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
