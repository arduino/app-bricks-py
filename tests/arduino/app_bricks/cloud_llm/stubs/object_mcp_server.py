# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import asynccontextmanager
import random

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP


def object_mcp_server_factory(port: int) -> FastAPI:
    mcp = FastMCP("home_automation", port=port)

    acStatuses = {}

    @mcp.tool()
    def light_control(state: bool, room: str) -> str:
        action = "turned on" if state else "turned off"
        return f"The light in the {room} has been {action}."

    @mcp.tool()
    def air_conditioner_control(state: bool, temperature: float, room: str) -> str:
        acStatuses[room] = state
        return (
            f"The air conditioner in the {room} has been turned on and set to {temperature}°C."
            if state
            else f"The air conditioner in the {room} has been turned off."
        )

    @mcp.tool()
    def get_room_temperature(room: str) -> int:
        return random.randint(18, 30)

    @mcp.tool()
    def get_ac_status(room: str) -> str:
        return "on" if acStatuses.get(room, False) else "off"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.mount("/", mcp.streamable_http_app())

    return app
