# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING
from collections.abc import Iterable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from arduino.app_utils import brick

if TYPE_CHECKING:
    import httpx


class HTTPEndpoint:
    """A class to communicate with remote MCP server via HTTP protocol to perform various tasks."""

    def __init__(self, name: str, url: str, headers: dict | None = None, token: str | None = None, auth: "httpx.Auth | None" = None) -> None:
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
        self.name = name
        self.config: dict = {"url": url}
        if headers:
            self.config["headers"] = headers
        if auth is not None:
            self.config["auth"] = auth

    def to_conn(self) -> dict:
        """Build the connection configuration consumed by MultiServerMCPClient.

        Returns:
            dict: A mapping of the endpoint name to its transport configuration.
        """
        return {
            self.name: {
                "transport": "http",
                **self.config,
            }
        }


@brick
class MCPClient:
    """A class to communicate with the MCP server to perform various tasks."""

    def __init__(self, endpoints: list[HTTPEndpoint], tool_name_prefix: bool = True, **kwargs) -> None:
        """Initialize the MCPClient with a MultiServerMCPClient.

        Args:
            endpoints (list[HTTPEndpoint]): A list of MCP endpoint configurations. Use the brick's HTTPEndpoint class
                to create endpoint configurations.
            tool_name_prefix (bool, optional): Whether to prefix tool names with the client name. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the MultiServerMCPClient.

        """
        connections = {}
        for endpoint in endpoints:
            connections.update(endpoint.to_conn())
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

    def list_tools(self) -> dict[str, str]:
        """Return a ``{tool_name: description}`` overview of the available tools.

        Useful for discovery: see what each MCP server exposes to decide which tools to keep via
        ``get_tools(include=..., exclude=...)``. Use ``inspect_tool`` for a single tool's details.

        Returns:
            dict[str, str]: Mapping of each tool's name to its description.
        """
        return {tool.name: tool.description for tool in self.get_tools()}

    def inspect_tool(self, name: str) -> dict | None:
        """Return the details of a single tool, or None if no tool has that name.

        Args:
            name (str): The tool name, as returned by ``list_tools`` / ``get_tools``.

        Returns:
            dict | None: ``{"name", "description", "parameters"}`` where ``parameters`` is the tool's
                argument schema, or None if no tool matches ``name``.
        """
        for tool in self.get_tools():
            if tool.name == name:
                return {"name": tool.name, "description": tool.description, "parameters": tool.args}
        return None
