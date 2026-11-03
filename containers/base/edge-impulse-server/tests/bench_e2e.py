# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""End-to-end benchmark against a running server: open a model, send random frames one at a time,
report fps, round-trip latency and the server's timing breakdown.

    docker exec <server container> python3 /repo/tests/bench_e2e.py /ipc/ei.sock <model> [frames]

Run it inside the server container (it has numpy) with the repository mounted at /repo, or anywhere
with numpy and access to the socket."""

import json
import os
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import protocol as P

sock_path, model, count = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
P.send_json(s, P.OPEN, {"model": model})
r = P.Reader(s)
kind, payload = r.read()
info = json.loads(bytes(payload))
if kind != P.OPENED:
    sys.exit(f"open failed: {info}")
w, h = info["width"], info["height"]
frames = [np.random.randint(0, 256, (h, w, 3), np.uint8) for _ in range(4)]
lat, timing = [], {}
for i in range(count + 5):
    t0 = time.perf_counter()
    P.send_frame(s, i, time.monotonic_ns(), frames[i % 4])
    kind, payload = r.read()
    dt = (time.perf_counter() - t0) * 1e3
    d = json.loads(bytes(payload))
    if kind != P.RESULT:
        sys.exit(f"error: {d}")
    if i >= 5:  # skip warm-up
        lat.append(dt)
        for k, v in d["timing_ms"].items():
            timing.setdefault(k, []).append(v)
lat.sort()
print(f"{model}: {w}x{h}, {len(lat)} frames, {1e3 / statistics.mean(lat):.1f} fps")
mean, p50, p95 = statistics.mean(lat), lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
print(f"  round trip ms: mean {mean:.1f}  p50 {p50:.1f}  p95 {p95:.1f}")
print("  server breakdown, mean ms: " + "  ".join(f"{k} {statistics.mean(v):.2f}" for k, v in timing.items()))
client = mean - statistics.mean(timing["recv"]) - statistics.mean(timing["server"])
print(f"  client side (round trip - recv - server): {client:.2f} ms")
