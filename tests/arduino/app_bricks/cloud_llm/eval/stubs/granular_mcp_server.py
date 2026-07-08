# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import random
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP


def granular_mcp_server_factory(port: int) -> FastAPI:
    mcp = FastMCP("home_automation", port=port)

    thermostat_temperatures = {}

    @mcp.tool()
    def turn_on_light(room: str) -> str:
        """Turn on the light in the specified room."""
        return f"The light in the {room} has been turned on."

    @mcp.tool()
    def turn_off_light(room: str) -> str:
        """Turn off the light in the specified room."""
        return f"The light in the {room} has been turned off."

    @mcp.tool()
    def set_thermostat(temperature: float, room: str) -> str:
        """Set the thermostat of the air conditioner to the specified temperature in the specified room."""
        thermostat_temperatures[room] = temperature
        return f"The thermostat has been set to {temperature}°C in the {room}."

    @mcp.tool()
    def get_thermostat(room: str) -> int:
        """Get the current thermostat temperature of the air conditioner in the specified room."""
        return thermostat_temperatures.get(room, 22)

    @mcp.tool()
    def turn_on_ac(room: str) -> str:
        """Turn on the air conditioner in the specified room."""
        return f"The air conditioner in the {room} has been turned on."

    @mcp.tool()
    def turn_off_ac(room: str) -> str:
        """Turn off the air conditioner in the specified room."""
        return f"The air conditioner in the {room} has been turned off."

    @mcp.tool()
    def get_temperature(room: str) -> float:
        """Get the current temperature in the specified room."""
        return round(random.uniform(18.0, 30.0), 2)

    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.mount("/", mcp_app)

    return app
