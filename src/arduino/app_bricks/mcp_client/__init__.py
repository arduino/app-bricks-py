# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
from abc import ABC, abstractmethod
from langchain_mcp_adapters.client import MultiServerMCPClient

from arduino.app_utils import brick


class MCPEndpoint(ABC):
    """A class representing an MCP endpoint configuration."""

    def __init__(self, name: str, transport: str, **kwargs):
        self.name = name
        self.transport = transport
        self.config = kwargs

    @abstractmethod
    def to_dict(self):
        config: dict = {}
        config[self.name] = {
            "transport": self.transport,
            **self.config,
        }
        return config


class HTTPEndpoint(MCPEndpoint):
    """A class to communicate with remote MCP server via HTTP protocol to perform various tasks."""

    def __init__(self, name: str, url: str, headers: dict = None):
        """Initialize the HTTPEndpoint with the given name, URL, and optional headers.
        Configure url to point to the /mcp endpoint of the remote MCP server.
        To add authentication, include the necessary headers (e.g., Authorization) in the headers dictionary.

        Args:
            name (str): A unique name for the MCP endpoint configuration.
            url (str): The URL of the remote MCP server's /mcp endpoint (e.g., http://localhost:8080/mcp).
            headers (dict, optional): Optional HTTP headers for authentication or other purposes. Defaults to None.
        """
        super().__init__(name=name, transport="http", url=url, headers=headers)

    def to_dict(self):
        return super().to_dict()


class LocalPythonMCPEndpoint(MCPEndpoint):
    """A class to communicate with a local Python MCP server to perform various tasks."""

    def __init__(self, name: str, script_path: str, args: list = None):
        """Initialize the LocalPythonMCPEndpoint with the given name, script path, and optional arguments.
        The script specified by script_path should implement an MCP server using the MCPServer class from the langchain_mcp_adapters library.

        Args:
            name (str): A unique name for the MCP endpoint configuration.
            script_path (str): The path to the Python script implementing the MCP server.
            args (list, optional): Additional command-line arguments to pass to the script. Defaults to None.

        !!! python "Example usage"
            ```python
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("MathServer")


            @mcp.tool()
            def add(a: int, b: int) -> int:
                '''Add two numbers'''
                return a + b


            if __name__ == "__main__":
                mcp.run(transport="stdio")
            ```

        """
        super().__init__(name=name, transport="stdio", command="python", args=[script_path] + (args or []))

    def to_dict(self):
        return super().to_dict()


@brick
class MCPClient:
    """A class to communicate with the MCP server to perform various tasks."""

    def __init__(self, clients: list[MCPEndpoint], tool_name_prefix: bool = True, **kwargs):
        """Initialize the MCPClient with a MultiServerMCPClient.

        Args:
            clients (list[MCPEndpoint]): A list of MCP endpoint configurations. Use brick's exposed endpoint classes like
                HTTPEndpoint or LocalPythonMCPEndpoint to create endpoint configurations.
            tool_name_prefix (bool, optional): Whether to prefix tool names with the client name. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the MultiServerMCPClient.

        """
        connections = {}
        for client in clients:
            connections.update(client.to_dict())
        self._client = MultiServerMCPClient(
            connections=connections,
            tool_name_prefix=tool_name_prefix,
            **kwargs,
        )

    def get_client(self) -> MultiServerMCPClient:
        """Get the underlying MultiServerMCPClient instance.

        Returns:
            MultiServerMCPClient: The underlying MCP client instance.
        """
        return self._client

    def list_tools(self) -> str:
        """List the available tools from the MCP server.

        Returns:
            str: A list of available tools.
        """
        return asyncio.run(self._client.get_tools())
