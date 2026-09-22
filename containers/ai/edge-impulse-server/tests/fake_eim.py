# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Fake .eim process: speaks the Edge Impulse runner protocol. Behaviour depends on its name (argv[1]).
The tests wrap it in one <name>.eim script per behaviour."""

import json
import os
import socket
import sys
import time

name, path = sys.argv[1], sys.argv[2]
if name == "broken":
    sys.stderr.write("error while loading shared libraries: libQnnHtp.so: cannot open shared object file\n")
    sys.exit(127)
if name == "hang":
    time.sleep(3600)
if name == "slow":
    time.sleep(1.0)  # long enough for a second connection to arrive during the load
w, h = (64, 96) if name == "portrait" else (96, 64)
gray = name == "gray"
tracker = name == "tracker"
# The threshold blocks a real .eim reports, with the values set_threshold changes
thresholds = [{"id": 12, "type": "object_detection", "min_score": 0.3}]
if tracker:
    thresholds.append({"id": 28, "type": "object_tracking", "max_age": 1, "min_hits": 3, "iou_threshold": 0.3})
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(1)
conn, _ = srv.accept()
calls = 0
buf = b""
while True:
    chunk = conn.recv(1 << 20)
    if not chunk:
        break
    buf += chunk
    try:
        msg = json.loads(buf)
    except ValueError:
        continue
    buf = b""
    reply = {"id": msg["id"], "success": True}
    if "hello" in msg:
        reply.update({
            "project": {"id": 1, "name": "proj-" + name, "owner": "test", "deploy_version": 1},
            "model_parameters": {
                "image_input_width": w,
                "image_input_height": h,
                "image_channel_count": 1 if gray else 3,
                "labels": ["a", "b"],
                "model_type": "object_detection",
                "image_resize_mode": "fit-shortest",
                "thresholds": thresholds,
                **({"has_object_tracking": True} if tracker else {}),
            },
        })
    elif "set_threshold" in msg:
        values = msg["set_threshold"]
        block = next((t for t in thresholds if t["id"] == values.get("id")), None)
        if block is None:
            reply = {"id": msg["id"], "success": False, "error": f"unknown threshold block {values.get('id')}"}
        else:
            block.update({k: v for k, v in values.items() if k != "id"})
    elif "classify" in msg:
        calls += 1
        feats = msg["classify"]
        if name == "stuck" and calls > 1:
            time.sleep(3600)
        if name == "late" and calls == 2:
            # replies just after the server gave up, so the reply is late but still arrives
            time.sleep(int(os.environ.get("EI_TEST_TIMEOUT", "4")) + 1)
        if name == "die" and calls == 2 and not os.path.exists(os.environ["EI_FAKE_DIE"]):
            open(os.environ["EI_FAKE_DIE"], "w").close()  # only the first process dies
            os._exit(3)
        if len(feats) != w * h:
            reply = {"id": msg["id"], "success": False, "error": f"expected {w * h} features, got {len(feats)}"}
        elif gray and any(((f >> 16) & 255) != (f & 255) or ((f >> 8) & 255) != (f & 255) for f in feats[:100]):
            reply = {"id": msg["id"], "success": False, "error": "features are not gray"}
        elif (name == "badwarm" and calls == 1) or (name == "flaky" and calls % 2 == 0):
            reply = {"id": msg["id"], "success": False, "error": f"{name} failure on call {calls}"}
            sys.stderr.write(f"{name}: simulated failure\n")
            sys.stderr.flush()
        else:
            time.sleep(float(os.environ.get("EI_FAKE_SLEEP", "0.02")))  # the inference time
            reply.update({
                "result": {
                    "bounding_boxes": [{"label": "a", "value": 0.9, "x": 1, "y": 2, "width": 3, "height": 4}],
                    # the thresholds this process holds, so a test can tell which values reached which instance
                    "classification": {"first_feature": feats[0], "min_score": thresholds[0]["min_score"]},
                },
                "timing": {"dsp": 1, "classification": 2, "anomaly": 0},
            })
    conn.sendall(json.dumps(reply).encode() + b"\x00")
