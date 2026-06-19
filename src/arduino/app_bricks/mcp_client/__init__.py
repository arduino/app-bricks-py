# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import sys
from abc import ABC
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Iterable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from arduino.app_utils import brick

if TYPE_CHECKING:
    import httpx


class MCPEndpoint(ABC):
    """A class representing an MCP endpoint configuration."""

    def __init__(self, name: str, transport: str, **kwargs):
        self.name = name
        self.transport = transport
        self.config = kwargs

    def to_conn(self) -> dict:
        """Build the connection configuration consumed by MultiServerMCPClient.

        Returns:
            dict: A mapping of the endpoint name to its transport configuration.
        """
        return {
            self.name: {
                "transport": self.transport,
                **self.config,
            }
        }


class HTTPEndpoint(MCPEndpoint):
    """A class to communicate with remote MCP server via HTTP protocol to perform various tasks."""

    def __init__(self, name: str, url: str, headers: dict | None = None, token: str | None = None, auth: "httpx.Auth | None" = None):
        """Initialize the HTTPEndpoint with the given name, URL, and optional authentication.
        Configure url to point to the /mcp endpoint of the remote MCP server.

        Authentication can be provided in three ways (see the brick README for provider recipes):
        - ``token``: a convenience for bearer auth, added as an ``Authorization: Bearer <token>`` header
          (e.g. a GitHub PAT or a Stripe restricted key).
        - ``headers``: arbitrary custom headers, for providers that use their own scheme
          (e.g. Datadog's ``DD-API-KEY`` / ``DD-APPLICATION-KEY``, or HTTP ``Basic`` auth).
        - ``auth``: an ``httpx.Auth`` object, for advanced or rotating-credential schemes (e.g. OAuth).

        An explicit ``Authorization`` entry in ``headers`` takes precedence over ``token``.

        Args:
            name (str): A unique name for the MCP endpoint configuration.
            url (str): The URL of the remote MCP server's /mcp endpoint (e.g., http://localhost:8080/mcp).
            headers (dict, optional): Optional HTTP headers for authentication or other purposes. Defaults to None.
            token (str, optional): Bearer token added as an ``Authorization: Bearer`` header. Defaults to None.
            auth (httpx.Auth, optional): An httpx authentication object passed through to the HTTP client. Defaults to None.
        """
        headers = dict(headers) if headers else {}
        if token:
            headers.setdefault("Authorization", f"Bearer {token}")
        config: dict = {"url": url}
        if headers:
            config["headers"] = headers
        if auth is not None:
            config["auth"] = auth
        super().__init__(name=name, transport="http", **config)


class LocalPythonMCPEndpoint(MCPEndpoint):
    """A class to communicate with a local Python MCP server to perform various tasks."""

    def __init__(self, name: str, script_path: str, args: list | None = None, env: dict | None = None):
        """Initialize the LocalPythonMCPEndpoint with the given name, script path, and optional arguments.
        The script specified by script_path should implement an MCP server using ``FastMCP`` from the ``mcp`` library (see the example below).

        Args:
            name (str): A unique name for the MCP endpoint configuration.
            script_path (str): The path to the Python script implementing the MCP server.
            args (list, optional): Additional command-line arguments to pass to the script. Defaults to None.
            env (dict, optional): Environment variables for the server process, e.g. to pass credentials/API keys. Defaults to None.

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
        config: dict = {"command": sys.executable, "args": [script_path] + (args or [])}
        if env:
            config["env"] = env
        super().__init__(name=name, transport="stdio", **config)


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
            connections.update(client.to_conn())
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

    def get_tools(self, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> list[BaseTool]:
        """Discover the tools exposed by the configured MCP servers.

        The returned tools are LangChain ``BaseTool`` instances that can be passed directly to the LLM
        bricks via their ``tools`` argument (e.g. ``CloudLLM(tools=mcp.get_tools())`` or
        ``LargeLanguageModel(tools=mcp.get_tools())``).

        Filtering by name lets you curate a small, relevant subset, useful for models with a small
        context window. Names are matched as fnmatch-style glob patterns, so both exact names and
        wildcards work (e.g. ``files_*``, ``*_read``).

        Args:
            include (Iterable[str], optional): Keep only tools whose name matches one of these patterns
                (exact names or globs). Defaults to None (keep all).
            exclude (Iterable[str], optional): Drop tools whose name matches one of these patterns
                (exact names or globs). Defaults to None.

        Returns:
            list[BaseTool]: The tools aggregated from every configured endpoint, after filtering.
        """
        tools = asyncio.run(self._client.get_tools())
        if include is not None:
            patterns = list(include)
            tools = [t for t in tools if any(fnmatchcase(t.name, p) for p in patterns)]
        if exclude:
            patterns = list(exclude)
            tools = [t for t in tools if not any(fnmatchcase(t.name, p) for p in patterns)]
        return tools

    def describe_tools(self) -> dict[str, str]:
        """Return a ``{tool_name: description}`` mapping of the available tools.

        Useful for discovery: inspect what each MCP server exposes (title and description) to
        decide which tools to keep via ``get_tools(include=..., exclude=...)``.

        Returns:
            dict[str, str]: Mapping of each tool's name (title) to its description.
        """
        return {tool.name: tool.description for tool in self.get_tools()}
