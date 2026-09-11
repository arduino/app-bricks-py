from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import os
import time

from iw_scanner import scan_access_points, get_wireless_interface

app = FastAPI(
    title="TPS Location Client Scanner",
    description="Scans nearby WiFi Access Points using iw and serves results via REST API.",
    version="1.0.0"
)

SCAN_CACHE_SECONDS = int(os.getenv("SCAN_CACHE_SECONDS", "10"))
_cached_results: List[Dict[str, str]] = []
_cached_timestamp_ms: int = 0
_start_time: float = time.time()


@app.get("/")
async def root():
    """Container info."""
    return {
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    interface = get_wireless_interface()
    return {
        "status": "healthy",
        "interface_detected": interface is not None,
        "interface": interface,
        "uptime": time.time() - _start_time
    }


@app.get("/scan")
async def get_access_points(
    interface: Optional[str] = None,
    force_refresh: bool = False
) -> JSONResponse:
    """
    Scan and return nearby WiFi Access Points.

    Query Parameters:
        interface: Wireless interface to use (auto-detects if not provided).
        force_refresh: Force a new scan, bypassing cache.
    """
    global _cached_results, _cached_timestamp_ms

    now_ms = int(time.time() * 1000)

    if (not force_refresh and
            _cached_results and
            (now_ms - _cached_timestamp_ms) < SCAN_CACHE_SECONDS * 1000):
        age_ms = now_ms - _cached_timestamp_ms
        return JSONResponse(content={
            "status": "success",
            "cached": True,
            "age_ms": age_ms,
            "timestamp_ms": _cached_timestamp_ms,
            "count": len(_cached_results),
            "access_points": _cached_results
        })

    scan_result = scan_access_points(interface=interface)

    if scan_result is None:
        raise HTTPException(
            status_code=500,
            detail="WiFi scan failed. Verify iw is installed and interface is available."
        )

    access_points, timestamp_ms = scan_result
    _cached_results = access_points
    _cached_timestamp_ms = timestamp_ms

    return JSONResponse(content={
        "status": "success",
        "cached": False,
        "age_ms": 0,
        "timestamp_ms": timestamp_ms,
        "count": len(access_points),
        "access_points": access_points
    })
