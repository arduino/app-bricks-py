# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the MCPClient brick.

The underlying ``MultiServerMCPClient`` is the single mock seam: endpoint tests
replace the class with a recording fake to assert the connection configuration
the brick builds, and tool tests replace its async ``get_tools`` with a scripted
one, so the brick's own logic (auth header assembly, endpoint merging, fnmatch
tool filtering, tool introspection) is exercised without any network access.
"""

from types import SimpleNamespace

import pytest

import arduino.app_bricks.mcp_client as mcp_client_module
from arduino.app_bricks.mcp_client import HTTPEndpoint, MCPClient


# --- Fakes & helpers ---------------------------------------------------------


class FakeMultiServerMCPClient:
    """Recording stand-in for langchain_mcp_adapters' MultiServerMCPClient."""

    def __init__(self, connections=None, **kwargs):
        self.connections = connections
        self.kwargs = kwargs


def _tool(name: str, description: str = "", args: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, args=args or {})


@pytest.fixture
def make_client(monkeypatch):
    """Build a real MCPClient over the recording fake, with scriptable tools."""

    def _make(endpoints=None, tools=None, **kwargs):
        monkeypatch.setattr(mcp_client_module, "MultiServerMCPClient", FakeMultiServerMCPClient)
        client = MCPClient(endpoints=endpoints or [], **kwargs)

        async def fake_get_tools():
            return list(tools or [])

        client._client.get_tools = fake_get_tools
        return client

    return _make


# --- HTTPEndpoint configuration ----------------------------------------------


def test_endpoint_conn_has_http_transport_and_url():
    conn = HTTPEndpoint(name="srv", url="http://host:8080/mcp").to_conn()

    assert conn == {"srv": {"transport": "http", "url": "http://host:8080/mcp"}}


def test_endpoint_token_becomes_bearer_header():
    conn = HTTPEndpoint(name="srv", url="http://host/mcp", token="tok-123").to_conn()

    assert conn["srv"]["headers"] == {"Authorization": "Bearer tok-123"}


def test_endpoint_explicit_authorization_header_wins_over_token():
    endpoint = HTTPEndpoint(
        name="srv",
        url="http://host/mcp",
        headers={"Authorization": "Basic abc"},
        token="ignored",
    )

    assert endpoint.to_conn()["srv"]["headers"]["Authorization"] == "Basic abc"


def test_endpoint_custom_headers_are_passed_verbatim():
    headers = {"DD-API-KEY": "k", "DD-APPLICATION-KEY": "a"}

    conn = HTTPEndpoint(name="srv", url="http://host/mcp", headers=headers).to_conn()

    assert conn["srv"]["headers"] == headers


def test_endpoint_does_not_mutate_caller_headers():
    headers = {"X-Custom": "1"}

    HTTPEndpoint(name="srv", url="http://host/mcp", headers=headers, token="tok")

    assert headers == {"X-Custom": "1"}


def test_endpoint_auth_object_is_passed_through():
    auth = object()

    conn = HTTPEndpoint(name="srv", url="http://host/mcp", auth=auth).to_conn()

    assert conn["srv"]["auth"] is auth


def test_endpoint_omits_unset_optionals():
    conn = HTTPEndpoint(name="srv", url="http://host/mcp").to_conn()

    assert "headers" not in conn["srv"]
    assert "auth" not in conn["srv"]


# --- MCPClient construction ---------------------------------------------------


def test_client_merges_endpoints_into_connections(make_client):
    client = make_client(
        endpoints=[
            HTTPEndpoint(name="alpha", url="http://a/mcp"),
            HTTPEndpoint(name="beta", url="http://b/mcp", token="tok"),
        ]
    )

    connections = client.get_client().connections
    assert set(connections) == {"alpha", "beta"}
    assert connections["beta"]["headers"] == {"Authorization": "Bearer tok"}


def test_client_forwards_prefix_flag_and_kwargs(make_client):
    client = make_client(endpoints=[], tool_name_prefix=False, custom_option=42)

    assert client.get_client().kwargs == {"tool_name_prefix": False, "custom_option": 42}


# --- get_tools filtering -------------------------------------------------------


@pytest.fixture
def tool_names():
    return ["files_read", "files_write", "math_add", "math_multiply"]


@pytest.mark.parametrize(
    "include, exclude, expected",
    [
        (None, None, ["files_read", "files_write", "math_add", "math_multiply"]),
        (["math_add"], None, ["math_add"]),
        (["files_*"], None, ["files_read", "files_write"]),
        (["*_read", "*_add"], None, ["files_read", "math_add"]),
        (None, ["files_*"], ["math_add", "math_multiply"]),
        (["math_*"], ["*_multiply"], ["math_add"]),
        (["nope_*"], None, []),
        ([], None, []),
    ],
)
def test_get_tools_filters_by_name_patterns(make_client, tool_names, include, exclude, expected):
    client = make_client(tools=[_tool(n) for n in tool_names])

    tools = client.get_tools(include=include, exclude=exclude)

    assert [t.name for t in tools] == expected


def test_get_tools_matching_is_case_sensitive(make_client):
    client = make_client(tools=[_tool("Files_Read")])

    assert client.get_tools(include=["files_*"]) == []
    assert [t.name for t in client.get_tools(include=["Files_*"])] == ["Files_Read"]


# --- list_tools / inspect_tool -------------------------------------------------


def test_list_tools_maps_names_to_descriptions(make_client):
    client = make_client(tools=[_tool("add", "Add two numbers"), _tool("echo", "Echo back")])

    assert client.list_tools() == {"add": "Add two numbers", "echo": "Echo back"}


def test_inspect_tool_returns_details(make_client):
    args_schema = {"a": {"type": "integer"}, "b": {"type": "integer"}}
    client = make_client(tools=[_tool("add", "Add two numbers", args_schema)])

    assert client.inspect_tool("add") == {
        "name": "add",
        "description": "Add two numbers",
        "parameters": args_schema,
    }


def test_inspect_tool_returns_none_when_missing(make_client):
    client = make_client(tools=[_tool("add")])

    assert client.inspect_tool("missing") is None
