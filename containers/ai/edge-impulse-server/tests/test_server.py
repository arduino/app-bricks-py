# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Functional tests of the real server against fake .eim processes (fake_eim.py).

Each fake behaviour (exiting, hanging, dying, replying late...) is a <name>.eim wrapper generated in a
temporary models directory. The server runs through run_server.py, with fake memory readings and short
timeouts. The tests share one running server and are ordered: the startup failures come first, the
shutdown and the low memory server last.

EI_TEST_DIR chooses where the fake .eim files are written, the default temporary directory is mounted
noexec in a hardened container. EI_TEST_TIMEOUT raises the server timeouts on a slow board.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import protocol as P
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# Runner start deadline and request timeout of the server under test (run_server.py)
TIMEOUT = int(os.environ.get("EI_TEST_TIMEOUT", "4"))
W, H = 96, 64
STD = ["--max-models", "2", "--max-clients", "3", "--memory-reserve-mb", "512"]
FAKES = [
    "det",
    "cls",
    "extra",
    "slow",
    "broken",
    "hang",
    "badwarm",
    "flaky",
    "stuck",
    "late",
    "die",
    "gray",
    "portrait",
    "noexec",
    "sub/dir/nested",
    "tracker",
]


class Harness:
    """A models directory of fake .eim files, the server under test and the client-side helpers."""

    def __init__(self, work):
        self.sock, self.models, self.log = f"{work}/ei.sock", f"{work}/models", f"{work}/server.log"
        self.conns = 0
        self.proc = None
        os.environ["EI_FAKE_DIE"] = f"{work}/die"
        os.mkdir(self.models)
        for name in FAKES:
            path = f"{self.models}/{name}.eim"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"#!{sys.executable}\nimport os, sys\n")  # no shell needed
                f.write(f"os.execv(sys.executable, [sys.executable, '{HERE}/fake_eim.py', '{name}', *sys.argv[1:]])\n")
            os.chmod(path, 0o644 if name == "noexec" else 0o755)
        if not os.access(f"{self.models}/det.eim", os.X_OK):
            pytest.fail(f"cannot execute the fake .eim files in {self.models}, the filesystem is probably mounted noexec: set EI_TEST_DIR")

    # server

    def start(self, *args, env=None, wait_socket=True):
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        self.proc = subprocess.Popen(
            [sys.executable, f"{HERE}/run_server.py", "--socket", self.sock, "--models-dir", self.models, "--stats-every", "1", *args],
            stderr=open(self.log, "w"),
            stdout=subprocess.DEVNULL,
            # EI_ACCEL=cpu so the tests also run inside an NPU image, which defaults to qnn mode
            env={**os.environ, "EI_ACCEL": "cpu", **(env or {})},
        )
        while wait_socket and not os.path.exists(self.sock) and self.proc.poll() is None:
            time.sleep(0.05)
        return self.proc

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(10)

    def log_text(self):
        return open(self.log).read()

    def last_line(self):
        return (self.log_text().strip().splitlines() or [""])[-1][:120]

    def server_traceback(self):
        return any("Traceback" in line and "ei.runner" not in line for line in self.log_text().splitlines())

    def eim_running(self, name=""):
        return self.eim_count(name) > 0

    def eim_count(self, name=""):
        """Running fake .eim processes of the model."""
        needle = f"fake_eim.py {name}"
        if os.path.isdir("/proc"):
            count = 0
            for pid in filter(str.isdigit, os.listdir("/proc")):
                try:
                    cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
                except OSError:
                    continue
                count += needle in cmd
            return count
        return len(subprocess.run(["pgrep", "-f", needle], capture_output=True).stdout.split())

    def wait_for(self, cond, timeout=TIMEOUT + 3):
        """True as soon as cond() holds, False after the timeout: terminating a fake .eim takes ms on a laptop, longer on a board."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.05)
        return cond()

    # client

    def connect(self):
        self.conns += 1
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock)
        s.settimeout(10)
        return s

    def raw(self, s, kind, payload):
        s.sendall(P.HEADER.pack(kind, len(payload)) + payload)

    def read(self, s):
        try:
            k, p = P.Reader(s).read()
            return k, json.loads(bytes(p))
        except (TimeoutError, ConnectionError) as e:
            return None, type(e).__name__

    def open_model(self, name):
        s = self.connect()
        P.send_json(s, P.OPEN, {"model": name})
        k, d = self.read(s)
        return s, k, d

    def frame(self, s, w=W, h=H, c=3, seq=1, fill=0, color=P.RGB):
        P.send_frame(s, seq, 123, np.full((h, w, c), fill, np.uint8), color)
        return self.read(s)


@pytest.fixture(scope="module")
def harness():
    work = tempfile.mkdtemp(prefix="ei-test-", dir=os.environ.get("EI_TEST_DIR") or None)
    h = Harness(work)
    yield h
    if h.proc is not None and h.proc.poll() is None:
        h.stop()
    shutil.rmtree(work, ignore_errors=True)


@pytest.fixture(scope="module")
def server(harness):
    """The standard server with det pinned, shared by the tests below and stopped by the shutdown test."""
    harness.start(*STD, "--pinned-models", " det, det,,", "--log-level", "INFO")
    return harness


# ---------------------------------------------------------------- startup


@pytest.mark.parametrize(
    ("args", "expect"),
    [
        pytest.param(["--max-models", "2", "--max-clients", "3"], "required", id="missing required arg"),
        pytest.param(["--max-mod", "2", "--max-clients", "3", "--memory-reserve-mb", "1"], "error", id="abbreviated arg rejected"),
        pytest.param(
            ["--max-models", "1", "--max-clients", "3", "--memory-reserve-mb", "1", "--pinned-models", "det,cls"],
            "Startup failed",
            id="pinned > max-models",
        ),
        pytest.param([*STD, "--pinned-models", "nope"], "not found", id="pinned missing model"),
        pytest.param([*STD, "--pinned-models", "broken"], "libQnnHtp.so", id="pinned broken model, .eim stderr in message"),
        pytest.param([*STD, "--pinned-models", "hang"], "did not start within", id="pinned model that never starts, deadline"),
        pytest.param([*STD, "--pinned-models", "noexec"], "not executable", id="pinned non-executable model"),
        pytest.param([*STD, "--accel", "qnn"], "fastrpc", id="qnn mode without fastrpc"),
    ],
)
def test_startup_fails(harness, args, expect):
    p = harness.start(*args, wait_socket=False)
    p.wait(15)
    assert p.returncode != 0
    assert expect in harness.log_text(), harness.last_line()


def test_startup_failure_after_a_pinned_load_stops_the_loaded_model(harness):
    p = harness.start(*STD, "--pinned-models", "det,broken", wait_socket=False)
    p.wait(15)
    time.sleep(0.3)
    assert p.returncode != 0, harness.last_line()
    assert not harness.eim_running("det")


def test_startup_pinned_load_below_memory_reserve_fails(harness):
    p = harness.start(*STD, "--pinned-models", "det", wait_socket=False, env={"FAKE_MEM_MB": "100"})
    p.wait(10)
    assert p.returncode != 0, harness.last_line()
    assert "available memory 100 MB" in harness.log_text()


# ---------------------------------------------------------------- running server


def test_pinned_model_loaded_at_startup(server):
    t = server.log_text()
    assert t.count("[det] loaded") == 1 and "(pinned)" in t, "duplicates and blanks in --pinned-models are ignored"
    assert server.eim_running("det")
    assert "CPU mode" in t and "max 2 models, 3 connections, reserve 512 MB" in t
    assert oct(os.stat(server.sock).st_mode & 0o777) == "0o660"


def test_open_and_infer(server):
    s, k, d = server.open_model("det")
    assert k == P.OPENED, d
    assert set(d) == {
        "model",
        "project",
        "width",
        "height",
        "channels",
        "labels",
        "model_type",
        "resize_mode",
        "object_tracking",
        "thresholds",
        "confidence",
        "slots",
    }
    assert d["slots"] == 1, "one frame in flight until the server runs more instances"
    assert d["object_tracking"] is False and d["thresholds"] == [{"id": 12, "type": "object_detection", "min_score": 0.3}]
    assert d["confidence"] is None, "without a confidence the connection receives everything the model reports"
    k, d = server.frame(s, fill=0)
    assert k == P.RESULT, d
    assert set(d) == {"seq", "ts_ns", "boxes", "tracks", "classes", "anomaly", "timing_ms", "slots"}
    assert d["tracks"] == [], "a model without object tracking reports no tracks"
    assert set(d["timing_ms"]) == {"recv", "resize", "encode", "lock", "inference", "dsp", "nn", "server"}
    P.send_frame(s, 7, 1, np.tile(np.array([10, 20, 30], np.uint8), (H, W, 1)))
    k, d = server.read(s)
    assert k == P.RESULT and d["classes"]["first_feature"] == (10 << 16) | (20 << 8) | 30, "RGB pixel encoded as (r<<16)|(g<<8)|b"
    P.send_frame(s, 8, 1, np.tile(np.array([10, 20, 30], np.uint8), (H, W, 1)), P.BGR)
    k, d = server.read(s)
    assert k == P.RESULT and d["classes"]["first_feature"] == (30 << 16) | (20 << 8) | 10, "a BGR frame is swapped to RGB"
    s.close()


def test_frames_of_any_size_are_resized_to_the_model_input(server):
    """The fake det model is 96x64 fit-shortest and always reports the box (1, 2, 3, 4) in model pixels."""
    s, k, d = server.open_model("det")
    k, d = server.frame(s, w=192, h=128, seq=2)
    assert k == P.RESULT, d
    box = d["boxes"][0]
    assert (box["x"], box["y"], box["w"], box["h"]) == (2, 4, 6, 8), "twice the model size, no crop: boxes scaled by 2"
    k, d = server.frame(s, w=32, h=32, seq=3)
    assert k == P.RESULT, ("smaller frames are upscaled", d)
    k, d = server.frame(s, w=640, h=480, seq=4)
    assert k == P.RESULT, d
    box = d["boxes"][0]
    assert abs(box["x"] - 1 / 0.15) < 0.1 and abs(box["w"] - 3 / 0.15) < 0.1, ("cropped to 640x427 then scaled by 0.15", box)
    assert abs(box["y"] - (26 + 2 * 427 / 64)) < 0.2, ("the crop offset is mapped back", box)
    s.close()


def test_a_tracking_model_reports_its_tracks_in_frame_coordinates(server):
    """The fake tracker reports the track (1, 2, 3, 4) of object 3 in model pixels."""
    s, k, d = server.open_model("tracker")
    assert k == P.OPENED, d
    assert d["object_tracking"] is True and [t["type"] for t in d["thresholds"]] == ["object_detection", "object_tracking"]
    k, d = server.frame(s, w=192, h=128, seq=1)
    assert k == P.RESULT, d
    assert d["tracks"] == [{"label": "a", "score": 0.9, "x": 2, "y": 4, "w": 6, "h": 8, "id": 3}], "mapped back like the boxes, with the object id"
    s.close()
    assert server.wait_for(lambda: not server.eim_running("tracker")), "released with its last connection"


def open_with_confidence(server, name, confidence):
    s = server.connect()
    P.send_json(s, P.OPEN, {"model": name, "confidence": confidence})
    k, d = server.read(s)
    return s, k, d


def test_each_connection_receives_what_reaches_its_confidence(server):
    """The fake det model reports one box scoring 0.9, and its score threshold as the anomaly score."""
    high, k, d = open_with_confidence(server, "det", 0.95)
    assert k == P.OPENED and d["confidence"] == 0.95, d
    assert d["thresholds"][0]["min_score"] == 0.95, "the threshold of the model follows its only connection"
    k, d = server.frame(high, seq=1)
    assert k == P.RESULT and d["boxes"] == [] and d["anomaly"] == 0.95, "nothing reaches 0.95, the .eim itself is at 0.95"
    low, k, d = open_with_confidence(server, "det", 0.5)
    assert k == P.OPENED and d["thresholds"][0]["min_score"] == 0.5, "the model goes down to the lowest confidence"
    k, d = server.frame(low, seq=1)
    assert k == P.RESULT and len(d["boxes"]) == 1 and d["anomaly"] == 0.5
    k, d = server.frame(high, seq=2)
    assert k == P.RESULT and d["boxes"] == [] and d["anomaly"] == 0.5, "the first connection still gets only what reaches its own"
    P.send_json(high, P.CONFIGURE, {"confidence": 0.2})
    k, d = server.read(high)
    assert k == P.CONFIGURE and d["confidence"] == 0.2 and d["thresholds"][0]["min_score"] == 0.2, "a lower confidence at runtime lowers the model"
    assert len(server.frame(high, seq=3)[1]["boxes"]) == 1
    low.close()
    assert server.wait_for(lambda: server.frame(high, seq=4)[1]["anomaly"] == 0.2), "the model follows the connections left"
    high.close()
    plain, k, d = server.open_model("det")
    assert k == P.OPENED and d["thresholds"][0]["min_score"] == 0.3, "back to the exported value when nobody asks for a confidence"
    assert server.frame(plain, seq=1)[1]["anomaly"] == 0.3
    plain.close()


def test_invalid_confidences_are_refused(server):
    for confidence in [1.5, -0.1, "high", True]:
        s = server.connect()
        P.send_json(s, P.OPEN, {"model": "det", "confidence": confidence})
        k, d = server.read(s)
        assert k == P.ERROR and d["code"] == "bad_request" and "confidence" in d["error"], (confidence, d)
        s.close()
    s, k, d = server.open_model("det")
    for values in [{"confidence": 2}, {"confidence": 0.5, "extra": 1}, {"other": 1}, {}]:
        P.send_json(s, P.CONFIGURE, values)
        k, d = server.read(s)
        assert k == P.ERROR and d["op"] == "configure" and d["code"] == "bad_request", (values, d)
    assert server.frame(s, seq=1)[0] == P.RESULT, "the connection stays open"
    s.close()


def test_thresholds_are_set_on_the_model_for_every_connection(server):
    s, k, d = server.open_model("tracker")
    P.send_json(s, P.CONFIGURE, {"id": 28, "max_age": 5})
    k, d = server.read(s)
    assert k == P.CONFIGURE and d["thresholds"][1] == {"id": 28, "type": "object_tracking", "max_age": 5, "min_hits": 3, "iou_threshold": 0.3}, d
    other, k, d = server.open_model("tracker")
    assert k == P.OPENED and d["thresholds"][1]["max_age"] == 5, "the blocks belong to the model, every connection sees the value"
    other.close()
    for values, expected in [
        ({"id": 99, "max_age": 1}, "no threshold block 99"),
        ({"id": 28, "min_score": 0.5}, "follows the confidence"),
        ({"id": 28, "threshold": 3}, "exposes max_age, min_hits, iou_threshold, not threshold"),
        ({"id": 28}, "nothing was given"),
        ({"id": 28, "max_age": "long"}, "must be numbers"),
        ({"id": 28, "max_age": True}, "must be numbers"),
    ]:
        P.send_json(s, P.CONFIGURE, values)
        k, d = server.read(s)
        assert k == P.ERROR and d["op"] == "configure" and d["code"] == "bad_request" and expected in d["error"], (values, d)
    server.raw(s, P.CONFIGURE, b"[1, 2]")
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_request", ("not a JSON object", d)
    assert server.frame(s, seq=2)[0] == P.RESULT, "the connection stays open"
    s.close()
    assert server.wait_for(lambda: not server.eim_running("tracker"))


def test_bad_frames_keep_the_connection_open(server):
    s, k, d = server.open_model("det")
    server.raw(s, P.FRAME, P.FRAME_HEADER.pack(2, 0, 4000, 3000, 3, 0) + b"\x00" * 10)
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_frame" and d["seq"] == 2, ("frame above the pixel limit", d)
    server.raw(s, P.FRAME, P.FRAME_HEADER.pack(3, 0, 1920, 1090, 3, 0) + b"\x00" * (1920 * 1090 * 3))
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_frame" and d["seq"] == 3, ("frame above the payload limit is discarded, not buffered", d)
    assert server.frame(s, seq=4)[0] == P.RESULT
    server.raw(s, P.FRAME, P.FRAME_HEADER.pack(9, 0, W, H, 3, 7) + b"\x00" * (W * H * 3))
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_frame", ("unknown color code", d)
    k, d = server.frame(s, c=1, seq=5)
    assert k == P.ERROR and d["code"] == "bad_frame", ("1 channel", d)
    server.raw(s, P.FRAME, P.FRAME_HEADER.pack(6, 0, W, H, 3, 0) + b"\x00" * 10)
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_frame" and d["seq"] == 6, ("truncated pixels", d)
    server.raw(s, P.FRAME, b"\x00" * 5)
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_frame" and d["seq"] is None, ("frame shorter than its header", d)
    s.close()


def test_unexpected_messages_keep_the_connection_open(server):
    s, k, d = server.open_model("det")
    P.send_json(s, P.OPEN, {"model": "cls"})
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_request", ("second OPEN", d)
    assert server.frame(s, seq=8)[0] == P.RESULT
    server.raw(s, b"XXXX", b"{}")
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_request", ("unknown type", d)
    s.sendall(P.HEADER.pack(b"XXXX", 2_000_000) + b"\x00" * 2_000_000)
    k, d = server.read(s)
    assert k == P.ERROR and d["code"] == "bad_request", ("oversize non-frame message", d)
    assert server.frame(s, seq=9)[0] == P.RESULT
    s.close()
    time.sleep(0.2)
    assert "[det] terminated" not in server.log_text() and server.eim_running("det"), "pinned model kept after its last user closes"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"[]", id="JSON array"),
        pytest.param(b"{not json", id="invalid JSON"),
        pytest.param(b'{"model": 5}', id="non-string model"),
        pytest.param(b"{}", id="no model"),
        pytest.param(b'"det"', id="string"),
        pytest.param(b"\x00" * 100_000, id="oversize"),
    ],
)
def test_invalid_open_closes_the_connection(server, payload):
    s = server.connect()
    server.raw(s, P.OPEN, payload)
    k, d = server.read(s)
    assert k == P.ERROR, (k, d)
    assert d["code"] == "bad_request"
    assert server.read(s)[0] is None, "connection closed"
    s.close()


def test_frame_before_open_closes_the_connection(server):
    s = server.connect()
    P.send_frame(s, 1, 0, np.zeros((H, W, 3), np.uint8))
    k, d = server.read(s)
    assert k == P.ERROR, (k, d)
    assert d["code"] == "bad_request"
    assert server.read(s)[0] is None, "connection closed"
    s.close()


@pytest.mark.parametrize("name", ["nope", "../det", ".hidden", "det/../det", "det;x", "sub/dir/../dir/nested", "/sub/dir/nested", "sub//dir/nested"])
def test_unknown_or_invalid_model_name_closes_the_connection(server, name):
    s, k, d = server.open_model(name)
    assert k == P.ERROR, (k, d)
    assert d["code"] == "unknown_model"
    assert server.read(s)[0] is None, "connection closed"
    s.close()


def test_payload_over_16_mib_closes_the_connection(server):
    s = server.connect()
    s.sendall(P.HEADER.pack(P.OPEN, 17 * 1024 * 1024))
    k, d = server.read(s)
    assert k is None, (k, d)
    s.close()
    time.sleep(0.2)
    assert not server.server_traceback(), "no traceback in the server log so far"


def test_max_models_admission(server):
    sc, k, d = server.open_model("cls")
    assert k == P.OPENED, ("second model loads", d)
    se, k, d = server.open_model("extra")
    assert k == P.ERROR and d["code"] == "too_many_models", ("third model", d)
    assert "cls" in d["error"] and "det" in d["error"], "the message lists the models in memory"
    sc.close()
    assert server.wait_for(lambda: "[cls] terminated" in server.log_text() and not server.eim_running("cls")), "cls terminated after its last close"
    se, k, d = server.open_model("extra")
    assert k == P.OPENED, ("slot freed, extra loads", d)
    se.close()
    time.sleep(0.3)


def test_max_clients_admission(server):
    held = [server.open_model("det")[0] for _ in range(3)]
    s, k, d = server.open_model("det")
    assert k == P.ERROR and d["code"] == "too_many_clients", ("4th connection", d)
    assert server.read(s)[0] is None, "connection closed"
    for h in held:
        h.close()
    time.sleep(0.2)


def test_concurrent_opens_share_one_load(server):
    before = server.log_text().count("[slow] loaded")
    out = {}
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, server.open_model("slow"))) for i in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    s3, k3, d3 = server.open_model("det")
    assert k3 == P.OPENED, ("an already loaded model does not wait for another load", d3)
    for t in threads:
        t.join()
    assert all(out[i][1] == P.OPENED for i in out), out
    assert server.log_text().count("[slow] loaded") - before == 1, "exactly one load"
    out[0][0].close()
    time.sleep(0.2)
    assert "[slow] terminated" not in server.log_text(), "model kept after the first of two closes"
    out[1][0].close()
    assert server.wait_for(lambda: "[slow] terminated" in server.log_text() and not server.eim_running("slow")), "terminated after the second close"
    s3.close()


def test_disconnect_during_load_completes_then_terminates(server):
    n_loaded, n_term = server.log_text().count("[slow] loaded"), server.log_text().count("[slow] terminated")
    s = server.connect()
    P.send_json(s, P.OPEN, {"model": "slow"})
    time.sleep(0.2)
    s.close()
    server.wait_for(lambda: server.log_text().count("[slow] terminated") == n_term + 1)
    t = server.log_text()
    assert t.count("[slow] loaded") == n_loaded + 1
    assert t.count("[slow] terminated") == n_term + 1


def test_load_failures_free_the_slot(server):
    s, k, d = server.open_model("broken")
    assert k == P.ERROR and d["code"] == "load_failed" and "libQnnHtp.so" in d["error"], ("exiting .eim, its stderr in the message", d)
    s.close()
    t0 = time.time()
    s, k, d = server.open_model("hang")
    dt = time.time() - t0
    assert k == P.ERROR and d["code"] == "load_failed" and "did not start" in d["error"], ("never-starting .eim", d)
    assert dt < TIMEOUT + 3, f"load_failed after the start deadline, took {dt:.1f}s"
    s.close()
    assert server.wait_for(lambda: not server.eim_running("hang")), "its process was killed"
    s, k, d = server.open_model("badwarm")
    assert k == P.ERROR and d["code"] == "load_failed", ("warm-up failure", d)
    s.close()
    assert server.wait_for(lambda: not server.eim_running("badwarm")), "its process was stopped"
    s, k, d = server.open_model("noexec")
    assert k == P.ERROR and d["code"] == "load_failed", ("non-executable .eim", d)
    s.close()
    s, k, d = server.open_model("cls")
    assert k == P.OPENED, ("slot freed after the failed loads", d)
    s.close()
    time.sleep(0.3)


def test_inference_error_keeps_the_model(server):
    s, k, d = server.open_model("flaky")
    k, d = server.frame(s, seq=1)
    assert k == P.ERROR and d["code"] == "internal" and "flaky failure" in d["error"], d
    assert server.frame(s, seq=2)[0] == P.RESULT, "connection open"
    time.sleep(0.2)
    assert "[flaky.eim] flaky: simulated failure" in server.log_text(), "the .eim stderr is logged with the model name"
    assert "[flaky] terminated" not in server.log_text()
    s.close()
    time.sleep(0.3)


def test_hung_eim_times_out_and_is_killed_when_the_client_leaves(server):
    s, k, d = server.open_model("stuck")
    t0 = time.time()
    k, d = server.frame(s, seq=1)
    dt = time.time() - t0
    assert k == P.ERROR and d["code"] == "internal", d
    assert dt < TIMEOUT + 3, f"internal after the request timeout, took {dt:.1f}s"
    s.close()
    time.sleep(0.5)
    assert not server.eim_running("stuck")


def test_late_reply_is_skipped(server):
    s, k, d = server.open_model("late")
    t0 = time.time()
    k, d = server.frame(s, seq=1)
    dt = time.time() - t0
    assert k == P.ERROR and d["code"] == "internal" and "no reply" in d["error"], d
    assert dt < TIMEOUT + 3, f"took {dt:.1f}s"
    time.sleep(2)  # let the late reply arrive before the next request
    k, d = server.frame(s, seq=2)
    assert k == P.RESULT and d["seq"] == 2, ("the next request gets its own reply", d)
    assert "skipped late reply" in server.log_text()
    s.close()
    time.sleep(0.3)


def test_dying_eim_is_restarted_in_place(server):
    s, k, d = server.open_model("die")
    k, d = server.frame(s, seq=1)
    assert k == P.ERROR and d["code"] == "internal" and "exited with code 3" in d["error"], d
    k, d = server.frame(s, seq=2)
    assert k == P.RESULT and d["seq"] == 2, ("next frame served", d)
    assert "restarting it" in server.log_text() and "[die] instance 2 replaces 1" in server.log_text(), "restart logged"
    s.close()
    assert server.wait_for(lambda: "[die] terminated" in server.log_text() and not server.eim_running("die")), "released after the client leaves"


def test_silent_connection_is_closed_after_the_open_timeout(server):
    s = server.connect()
    t0 = time.time()
    k, d = server.read(s)
    dt = time.time() - t0
    assert k == P.ERROR and d["code"] == "bad_request" and "no OPEN" in d["error"], d
    assert server.read(s)[0] is None, "connection closed"
    assert dt < 3, f"took {dt:.1f}s"
    s.close()


def test_model_in_a_subdirectory(server):
    s, k, d = server.open_model("sub/dir/nested")
    assert k == P.OPENED and d["model"] == "sub/dir/nested", d
    assert server.frame(s)[0] == P.RESULT
    s.close()
    assert server.wait_for(lambda: "[sub/dir/nested] terminated" in server.log_text())


def test_grayscale_and_portrait_models(server):
    s, k, d = server.open_model("gray")
    k, d = server.frame(s, fill=77)
    assert k == P.RESULT and d["classes"]["first_feature"] == 77 * 0x010101, ("gray features accepted by the .eim", d)
    s.close()
    time.sleep(0.3)
    s, k, d = server.open_model("portrait")
    assert k == P.OPENED and d["width"] == 64, d
    k2, d2 = server.frame(s, w=64, h=96)
    assert k2 == P.RESULT, ("exact-size frame accepted, no crop", d2)
    s.close()
    time.sleep(0.3)


def test_log_after_the_session(server):
    hc = server.connect()
    hc.close()
    time.sleep(0.3)
    assert f"[conn-{server.conns}]" not in server.log_text(), "connect and close without OPEN leaves no log entry"
    time.sleep(1.2)
    assert "connections 0/3 | models 1/2: det (0 conn, pinned)" in server.log_text(), "periodic stats line"
    errors = [line for line in server.log_text().splitlines() if "Error" in line]
    assert not server.server_traceback(), errors[:1]


def test_sigterm_stops_the_pinned_model_and_removes_the_socket(server):
    server.stop()
    time.sleep(0.3)
    assert "Stopped" in server.log_text()
    assert not os.path.exists(server.sock)
    assert not server.eim_running("det")


def test_open_below_the_memory_reserve(harness):
    harness.start(*STD, env={"FAKE_MEM_MB": "100"})
    s, k, d = harness.open_model("cls")
    assert k == P.ERROR and d["code"] == "low_memory", d
    assert "100 MB" in d["error"] and "512 MB" in d["error"], "the message has the details"
    s.close()
    harness.stop()
    assert not harness.eim_running(), "no fake .eim process left"


# ---------------------------------------------------------------- instances


@pytest.fixture(scope="module")
def scaled(harness):
    """A server allowed two instances per model, with fake .eim files taking 100 ms per inference."""
    harness.start(*STD, "--max-model-instances", "2", "--log-level", "INFO", env={"EI_FAKE_SLEEP": "0.1"})
    yield harness
    harness.stop()


def hammer(harness, s, until, timeout=8.0, start_seq=1):
    """Send frames one at a time, each as soon as its reply is in, until `until(kind, data)` holds.
    Returns the replies and the last sequence number sent, whose result may still be on its way."""
    replies, seq, deadline = [], start_seq, time.monotonic() + timeout
    P.send_frame(s, seq, 0, np.zeros((H, W, 3), np.uint8))
    while time.monotonic() < deadline:
        k, d = harness.read(s)
        replies.append((k, d))
        if until(k, d):
            return replies, seq
        if k == P.RESULT:
            seq += 1
            P.send_frame(s, seq, 0, np.zeros((H, W, 3), np.uint8))
    pytest.fail(f"condition not met, last replies: {replies[-3:]}")


def test_saturated_model_gets_a_second_instance_and_the_connection_a_second_slot(scaled):
    s, k, d = scaled.open_model("det")
    assert k == P.OPENED and d["slots"] == 1
    replies, last = hammer(scaled, s, lambda k, d: d.get("slots") == 2)
    assert scaled.wait_for(lambda: scaled.eim_count("det") == 2), "a second .eim process runs"
    assert "[det] instance 2 added" in scaled.log_text()
    assert any(k == P.SLOTS for k, _ in replies), "the second slot is granted with a SLOT message, half a period after a result"
    # Two frames in flight: the results arrive in order, and one that would arrive out of order is dropped instead
    P.send_frame(s, last + 1, 0, np.zeros((H, W, 3), np.uint8))
    P.send_frame(s, last + 2, 0, np.zeros((H, W, 3), np.uint8))
    answered, deadline = [], time.monotonic() + 3
    s.settimeout(0.5)
    while time.monotonic() < deadline:
        try:
            k, d = P.Reader(s).read()
        except TimeoutError:
            continue
        d = json.loads(bytes(d))
        assert k in (P.RESULT, P.SLOTS), (k, d)
        if k == P.RESULT and d["seq"] > last:
            answered.append(d["seq"])
    s.settimeout(10)
    assert answered and answered == sorted(answered) and answered[-1] == last + 2, answered
    s.close()
    time.sleep(0.3)
    assert scaled.wait_for(lambda: scaled.eim_count("det") == 0), "both instances go with the last connection"


def test_thresholds_reach_the_instances_added_later(scaled):
    s, k, d = open_with_confidence(scaled, "det", 0.55)
    assert k == P.OPENED and d["thresholds"][0]["min_score"] == 0.55
    replies, last = hammer(scaled, s, lambda k, d: d.get("slots") == 2)
    assert scaled.wait_for(lambda: scaled.eim_count("det") == 2)
    seen = {d["anomaly"] for k, d in replies if k == P.RESULT}
    for round in range(4):  # two frames in flight run on both instances
        P.send_frame(s, last + 2 * round + 1, 0, np.zeros((H, W, 3), np.uint8))
        P.send_frame(s, last + 2 * round + 2, 0, np.zeros((H, W, 3), np.uint8))
        for _ in range(2):
            k, d = scaled.read(s)
            while k != P.RESULT:
                k, d = scaled.read(s)
            seen.add(d["anomaly"])
    assert seen == {0.55}, "the value set before the second instance existed holds on it too"
    s.close()
    assert scaled.wait_for(lambda: scaled.eim_count("det") == 0)


def test_a_client_slower_than_the_model_gets_no_second_instance(scaled):
    s, k, d = scaled.open_model("cls")
    for seq in range(1, 8):
        k, d = scaled.frame(s, seq=seq)
        assert k == P.RESULT and d["slots"] == 1, d
        time.sleep(0.3)  # the model is idle three quarters of the time
    assert scaled.eim_count("cls") == 1 and "[cls] instance 2" not in scaled.log_text()
    s.close()


def test_idle_instance_is_retired_and_the_slot_taken_back(scaled):
    s, k, d = scaled.open_model("det")
    _, last = hammer(scaled, s, lambda k, d: d.get("slots") == 2)
    assert scaled.wait_for(lambda: scaled.eim_count("det") == 2)
    deadline = time.monotonic() + 8
    seq, slots = last + 1, 2
    while time.monotonic() < deadline and (slots == 2 or scaled.eim_count("det") == 2):
        time.sleep(0.4)  # the two instances are idle most of the time
        k, d = scaled.frame(s, seq=seq)
        seq += 1
        while k != P.RESULT or d["seq"] != seq - 1:  # SLOT messages and the result hammer() left in flight
            k, d = scaled.read(s)
        slots = d["slots"]
    assert slots == 1 and scaled.eim_count("det") == 1, "the second instance is retired and the client is back to one frame in flight"
    assert "[det] instance 2 retired" in scaled.log_text()
    s.close()


def test_a_model_keeping_state_between_frames_never_gets_a_second_instance(scaled):
    s, k, d = scaled.open_model("tracker")
    assert k == P.OPENED
    assert scaled.wait_for(lambda: "[tracker] single instance: object tracking" in scaled.log_text())
    replies, _ = hammer(scaled, s, lambda k, d: k == P.RESULT and d["seq"] >= 25)  # saturated for longer than the scale-up delay
    assert all(d["slots"] == 1 for k, d in replies), "two trackers would disagree on the object identities"
    assert scaled.eim_count("tracker") == 1 and "[tracker] instance 2" not in scaled.log_text()
    s.close()
