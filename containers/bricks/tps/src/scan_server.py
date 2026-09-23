# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""HTTP API serving Wi-Fi scan results to the app container over a Unix domain socket."""

import logging
import os
import threading
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from iw_scanner import ScanError, ScanResult, list_interfaces, scan

logger = logging.getLogger("scan_server")

SCAN_CACHE_SECONDS = int(os.getenv("SCAN_CACHE_SECONDS", "10"))


class ScanCache:
    """Serializes scans and reuses a result for ttl_seconds, which also bounds the scan rate."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_ms = ttl_seconds * 1000
        self._lock = threading.Lock()
        self._result: ScanResult | None = None

    def get(self) -> tuple[ScanResult, bool]:
        """Return the current result and whether it was served from cache."""
        with self._lock:
            now_ms = int(time.time() * 1000)
            if self._result is not None and now_ms - self._result.timestamp_ms < self._ttl_ms:
                return self._result, True
            self._result = scan()
            return self._result, False


app = FastAPI(title="TPS Location Wi-Fi Scanner", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
cache = ScanCache(SCAN_CACHE_SECONDS)


@app.get("/health")
def health() -> dict:
    """Report whether a wireless interface is available."""
    try:
        interfaces = list_interfaces()
    except ScanError as e:
        logger.error("health check failed: %s", e)
        raise HTTPException(status_code=503, detail="wireless interfaces unavailable")
    return {"status": "ok", "interfaces": interfaces}


@app.get("/scan")
def get_access_points() -> dict:
    """Return the nearby access points, from cache when recent enough."""
    try:
        result, cached = cache.get()
    except ScanError as e:
        logger.error("scan failed: %s", e)
        raise HTTPException(status_code=503, detail="Wi-Fi scan failed")

    return {
        "timestamp_ms": result.timestamp_ms,
        "age_ms": max(0, int(time.time() * 1000) - result.timestamp_ms),
        "cached": cached,
        "count": len(result.access_points),
        "access_points": [asdict(ap) for ap in result.access_points],
    }
