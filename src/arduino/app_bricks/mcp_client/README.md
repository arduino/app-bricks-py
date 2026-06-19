# MCP Client Brick

The MCP Client Brick connects your Arduino app to one or more [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers and exposes their tools as LangChain tools. Those tools can be passed straight to the LLM bricks (`Cloud LLM`, `LLM`), letting a model call external capabilities — file access, math, web services, your own MCP server — during a conversation.

## Overview

[MCP](https://modelcontextprotocol.io) is an open protocol that lets applications expose *tools* (callable functions) to AI models in a standard way. This Brick acts as an MCP **client**: you declare one or more endpoints (a local Python MCP server, or a remote MCP server reachable over HTTP) and the Brick discovers the tools they expose. The discovered tools are ready to be handed to an LLM Brick through its `tools` argument.

## Features

- **Multiple servers at once**: Aggregate tools from several MCP servers through a single client.
- **Local and remote transports**: Run a local Python MCP server over `stdio`, or connect to a remote server over HTTP.
- **Drop-in LLM integration**: `get_tools()` returns LangChain tools compatible with `CloudLLM(tools=...)` and `LargeLanguageModel(tools=...)`.
- **Tool name prefixing**: Optionally prefix tool names with the endpoint name to avoid clashes between servers.
- **Authentication**: Authenticate to remote servers with a bearer `token`, custom `headers`, or any `httpx.Auth`.

## Prerequisites

- **Python dependency**: Install the Brick extra:
  ```bash
  pip install arduino_app_bricks[mcp_client]
  ```
- **An MCP server**:
  - For a local endpoint, a Python script implementing an MCP server (e.g. with `FastMCP`). See `examples/math_server.py`.
  - For an HTTP endpoint, a reachable MCP server. The `examples/Dockerfile` shows how to expose a filesystem MCP server over HTTP using `mcp-proxy`.

## Endpoint Types

| Class | Transport | Parameters |
| :---- | :-------- | :--------- |
| `LocalPythonMCPEndpoint` | `stdio` | `name`: unique endpoint name. `script_path`: path to the Python MCP server script. `args`: optional extra CLI args. `env`: optional environment variables (e.g. credentials) for the server process. |
| `HTTPEndpoint` | HTTP | `name`: unique endpoint name. `url`: the remote server's `/mcp` endpoint. `headers`: optional custom HTTP headers. `token`: optional bearer token. `auth`: optional `httpx.Auth` for advanced/OAuth schemes. |

## Code Example and Usage

### Discover tools from a local MCP server

```python
from arduino.app_bricks.mcp_client import MCPClient, LocalPythonMCPEndpoint
from arduino.app_utils import App

local = LocalPythonMCPEndpoint(name="math", script_path="math_server.py")
mcp = MCPClient(clients=[local])

print(mcp.get_tools())  # -> list[BaseTool]

App.run()
```

### Give MCP tools to an LLM (the main use case)

`get_tools()` returns LangChain tools that plug directly into the LLM bricks. The model can then call the MCP tools while chatting.

```python
from arduino.app_bricks.mcp_client import MCPClient, LocalPythonMCPEndpoint
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

mcp = MCPClient(clients=[LocalPythonMCPEndpoint(name="math", script_path="math_server.py")])

llm = CloudLLM(
    model="google:gemini-2.5-flash",
    api_key="YOUR_API_KEY",  # Recommended: set it via the Brick Configuration in App Lab
    tools=mcp.get_tools(),
)

def ask():
    print(llm.chat("How much is 3 multiplied by 7?"))

App.run(ask)
```

The same `tools=mcp.get_tools()` also works with the on-device `LargeLanguageModel` brick.

### Connect to a remote MCP server over HTTP

```python
from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_utils import App

remote = HTTPEndpoint(name="filesystem", url="http://localhost:8080/mcp")
mcp = MCPClient(clients=[remote])

print(mcp.get_tools())

App.run()
```

See `examples/Dockerfile` for a ready-to-run HTTP MCP server (a filesystem server behind `mcp-proxy`).

## Authentication

Remote MCP servers authenticate clients with **static credentials sent in HTTP headers** — typically a bearer token, or provider-specific custom headers. `HTTPEndpoint` offers three ways to supply them:

| Way | Use it for | Result |
| :-- | :--------- | :----- |
| `token=...` | A bearer token / API key / PAT (GitHub, Stripe, …) | Adds an `Authorization: Bearer <token>` header |
| `headers={...}` | Provider-specific header schemes (Datadog, HTTP Basic, …) | Sends the headers verbatim |
| `auth=<httpx.Auth>` | Advanced or rotating credentials (e.g. OAuth) | Passed through to the underlying HTTP client |

An explicit `Authorization` entry in `headers` takes precedence over `token`. Keep secrets out of source — read them from environment variables (declared as **secret** Brick Configuration variables, see [Storing credentials](#storing-credentials)).

### Provider recipes

**Bearer token** (GitHub PAT, Stripe restricted key):

```python
import os
from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint

github = HTTPEndpoint(
    name="github",
    url="https://api.githubcopilot.com/mcp/",
    token=os.getenv("GITHUB_MCP_PAT"),
)
mcp = MCPClient(clients=[github])
```

**Custom headers** (Datadog uses two custom headers instead of a bearer token):

```python
import os
from arduino.app_bricks.mcp_client import HTTPEndpoint

datadog = HTTPEndpoint(
    name="datadog",
    url="https://<your-datadog-mcp-domain>/mcp",
    headers={
        "DD-API-KEY": os.getenv("DD_API_KEY", ""),
        "DD-APPLICATION-KEY": os.getenv("DD_APP_KEY", ""),
    },
)
```

**HTTP Basic** (e.g. Atlassian email + API token):

```python
import base64, os
from arduino.app_bricks.mcp_client import HTTPEndpoint

basic = base64.b64encode(f"{os.getenv('EMAIL')}:{os.getenv('API_TOKEN')}".encode()).decode()
atlassian = HTTPEndpoint(name="atlassian", url="https://...", headers={"Authorization": f"Basic {basic}"})
```

**Advanced / rotating credentials**: pass any [`httpx.Auth`](https://www.python-httpx.org/advanced/authentication/) via `auth=...` (e.g. an OAuth provider from the `mcp` SDK's `mcp.client.auth`). The object is passed through to the HTTP client, so it can refresh credentials per request.

**Local (stdio) servers** receive credentials through the process environment:

```python
import os
from arduino.app_bricks.mcp_client import LocalPythonMCPEndpoint

local = LocalPythonMCPEndpoint(
    name="myserver",
    script_path="myserver.py",
    env={"API_KEY": os.getenv("MY_API_KEY", "")},
)
```

### Storing credentials

Declare each credential as a **secret** variable in this brick's `brick_config.yaml`, so it can be set from the App Lab UI and read at runtime via `os.getenv`:

```yaml
variables:
  - name: GITHUB_MCP_PAT
    description: GitHub MCP server personal access token
    secret: true
```

Use one variable per credential (multiple servers/providers → multiple variables).

## API

| Member | Description |
| :----- | :---------- |
| `MCPClient(clients, tool_name_prefix=True, **kwargs)` | Create a client over a list of endpoints. When `tool_name_prefix` is `True`, tool names are prefixed with the endpoint name. Extra `kwargs` are forwarded to the underlying `MultiServerMCPClient`. |
| `MCPClient.get_tools(include=None, exclude=None) -> list[BaseTool]` | Discover and return the tools from all configured endpoints, as LangChain tools for the LLM bricks. Optionally filter by tool name (exact names or glob patterns) via `include`/`exclude` — handy to curate a small subset for small-context models. |
| `MCPClient.list_tools() -> dict[str, str]` | Return a `{tool_name: description}` overview of the available tools — useful to explore a server and decide what to `include`/`exclude`. |
| `MCPClient.inspect_tool(name) -> dict \| None` | Return one tool's details (`{name, description, parameters}`, where `parameters` is its argument schema), or `None` if not found. |
| `MCPClient.get_client() -> MultiServerMCPClient` | Access the underlying `langchain-mcp-adapters` client for advanced use (async sessions, prompts, resources). |
| `LocalPythonMCPEndpoint(name, script_path, args=None, env=None)` | A local Python MCP server launched over `stdio`. `env` passes environment variables (e.g. credentials) to the process. |
| `HTTPEndpoint(name, url, headers=None, token=None, auth=None)` | A remote MCP server reached over HTTP. `token` (bearer), `headers` (custom), and `auth` (`httpx.Auth`) configure authentication. |

## Notes

- **Sessionless by design**: `get_tools()` opens a fresh connection to each server when it is invoked and when a tool runs; there is no long-lived session to manage, so the Brick needs no explicit start/stop.
- For advanced scenarios (persistent sessions, prompts, resources), use `get_client()` to work with the underlying `MultiServerMCPClient` directly.
