# MCP Client Brick

The MCP Client Brick connects your Arduino app to one or more [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers and exposes their tools as LangChain tools. Those tools can be passed straight to the LLM bricks (`Cloud LLM`, `LLM`), letting a model call external capabilities — file access, math, web services, your own MCP server — during a conversation.

## Overview

[MCP](https://modelcontextprotocol.io) is an open protocol that lets applications expose *tools* (callable functions) to AI models in a standard way. This Brick acts as an MCP **client**: you declare one or more MCP servers reachable over HTTP and the Brick discovers the tools they expose. The discovered tools are ready to be handed to an LLM Brick through its `tools` argument.

## Features

- **Multiple servers at once**: Aggregate tools from several MCP servers through a single client.
- **Drop-in LLM integration**: `get_tools()` returns LangChain tools compatible with `CloudLLM(tools=...)` and `LargeLanguageModel(tools=...)`.
- **Tool name prefixing**: Optionally prefix tool names with the endpoint name to avoid clashes between servers.
- **Authentication**: Authenticate to servers with a bearer `token`, custom `headers`, or any `httpx.Auth`.

## Prerequisites

- **Python dependency**: Install the Brick extra:
  ```bash
  pip install arduino_app_bricks[mcp_client]
  ```
- **An MCP server**: a server exposing an HTTP `/mcp` endpoint, reachable from the app. You can also deploy one *together with the app* as a custom brick — see [Bundle an MCP server with your app](#bundle-an-mcp-server-with-your-app).

## Code Example and Usage

### Discover tools from an MCP server

```python
from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_utils import App

server = HTTPEndpoint(name="server", url="http://<mcp-server-host>:8080/mcp")
mcp = MCPClient(endpoints=[server])

print(mcp.get_tools())  # -> list[BaseTool]

App.run()
```

`name` is a unique label for the endpoint: by default it prefixes the discovered tool names (`server_<tool>`) to avoid clashes between servers. The URL points to the server's `/mcp` endpoint.

### Give MCP tools to an LLM (the main use case)

`get_tools()` returns LangChain tools that plug directly into the LLM bricks. The model can then call the MCP tools while chatting.

```python
from arduino.app_bricks.mcp_client import MCPClient, HTTPEndpoint
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

mcp = MCPClient(endpoints=[HTTPEndpoint(name="clock", url="http://mcp-server:8080/mcp")])

llm = CloudLLM(
    model="google:gemini-2.5-flash",
    api_key="YOUR_API_KEY",  # Recommended: set it via the Brick Configuration in App Lab
    tools=mcp.get_tools(),
)

def ask():
    print(llm.chat("What time is it in Rome?"))

App.run(ask)
```

Here the model answers by calling the `get_current_datetime` tool of the MCP server bundled with the app (see [Bundle an MCP server with your app](#bundle-an-mcp-server-with-your-app) below); `examples/02_with_custom_brick_and_llm` is the complete, runnable version. The same `tools=mcp.get_tools()` also works with the on-device `LargeLanguageModel` brick.

### Bundle an MCP server with your app

An app can ship its own MCP server as a **custom brick backed by a Docker container**: declare the container in a local brick's `brick_compose.yaml`, and App Lab deploys it automatically alongside the app — no external setup needed. The container can be a **prebuilt image** — `examples/01_with_custom_brick_basic_usage` runs the [Docker MCP Gateway](https://hub.docker.com/r/docker/mcp-gateway), which serves the MCP servers of the [Docker MCP catalog](https://hub.docker.com/catalogs/mcp) over HTTP — or an image **built from a Dockerfile shipped with the brick**, as in `examples/02_with_custom_brick_and_llm`:

```
my-app/
├── app.yaml                     # references the local brick by id
├── python/main.py
└── bricks/
    └── mcp-server/
        ├── brick_config.yaml    # id, name, description
        ├── brick_compose.yaml   # Docker Compose service running the server
        ├── Dockerfile           # image built for the service
        └── server.py            # the MCP server (FastMCP, streamable HTTP)
```

`app.yaml` lists the local brick next to this one:

```yaml
bricks:
  - arduino:mcp_client
  - mcp-server
```

All the app's containers share one network, so the Python code reaches the server using the **compose service name as hostname**:

```python
mcp = MCPClient(endpoints=[HTTPEndpoint(name="demo", url="http://mcp-server:8080/mcp")])
```

Declare a `healthcheck` in `brick_compose.yaml` so the app only starts once the server is ready.

## Authentication

MCP servers authenticate clients with **static credentials sent in HTTP headers** — typically a bearer token, or provider-specific custom headers. `HTTPEndpoint` offers three ways to supply them:

| Way | Use it for | Result |
| :-- | :--------- | :----- |
| `token=...` | A bearer token / API key / PAT (GitHub, Stripe, …) | Adds an `Authorization: Bearer <token>` header |
| `headers={...}` | Provider-specific header schemes (Datadog, HTTP Basic, …) | Sends the headers verbatim |
| `auth=<httpx.Auth>` | Advanced or rotating credentials (e.g. OAuth) | Passed through to the underlying HTTP client |

An explicit `Authorization` entry in `headers` takes precedence over `token`. Keep secrets out of source — read them from environment variables, set as brick variables in your `app.yaml` (see [Storing credentials](#storing-credentials)).

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
mcp = MCPClient(endpoints=[github])
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

### Storing credentials

Set each credential in your app's `app.yaml`, and retrieve them via `os.getenv`:

```yaml
bricks:
  - arduino:mcp_client:
      variables:
        GITHUB_MCP_PAT: "<your token>"
```

```python
github = HTTPEndpoint(
    name="github",
    url="https://<github>/mcp/",
    token=os.getenv("GITHUB_MCP_PAT"),
)
```

## API

| Member | Description |
| :----- | :---------- |
| `MCPClient(endpoints, tool_name_prefix=True, **kwargs)` | Create a client over a list of endpoints. When `tool_name_prefix` is `True`, tool names are prefixed with the endpoint name. Extra `kwargs` are forwarded to the underlying `MultiServerMCPClient`. |
| `MCPClient.get_tools(include=None, exclude=None) -> list[BaseTool]` | Discover and return the tools from all configured endpoints, as LangChain tools for the LLM bricks. Optionally filter by tool name (exact names or glob patterns) via `include`/`exclude` — handy to curate a small subset for small-context models. |
| `MCPClient.list_tools() -> dict[str, str]` | Return a `{tool_name: description}` overview of the available tools — useful to explore a server and decide what to `include`/`exclude`. |
| `MCPClient.inspect_tool(name) -> dict \| None` | Return one tool's details (`{name, description, parameters}`, where `parameters` is its argument schema), or `None` if not found. |
| `MCPClient.get_client() -> MultiServerMCPClient` | Access the underlying `langchain-mcp-adapters` client for advanced use (async sessions, prompts, resources). |
| `HTTPEndpoint(name, url, headers=None, token=None, auth=None)` | An MCP server reached over HTTP. `token` (bearer), `headers` (custom), and `auth` (`httpx.Auth`) configure authentication. |

## Notes

- **Sessionless by design**: `get_tools()` opens a fresh connection to each server when it is invoked and when a tool runs; there is no long-lived session to manage, so the Brick needs no explicit start/stop.
- For advanced scenarios (persistent sessions, prompts, resources), use `get_client()` to work with the underlying `MultiServerMCPClient` directly.
