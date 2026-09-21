# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""One running .eim process, driven through the Edge Impulse Linux SDK.

The SDK (edge_impulse_linux 1.2.2) is used for the runner protocol, but its
process handling needs a few fixes for a long-running server:
  - init() loops forever if the .eim exits before creating its socket:
    here the start has a deadline and a dead process is reported at once;
  - the runner socket has no timeout when shared memory is in use, so a
    hung .eim would block its model forever: here every request has one;
  - after a timeout the SDK reads the late reply as the answer to the next
    request and stays one reply behind forever: here late replies are skipped;
  - a .eim that dies mid-life makes the SDK fail with an IndexError: here
    its exit is reported as RunnerExited, with the code;
  - large feature messages are sent with send() instead of sendall();
  - stop() sends SIGINT without waiting: the process could survive or
    linger as a zombie;
  - get_features_from_image() is a per-pixel Python loop and crops portrait
    inputs even when the frame already has the model size: features are
    encoded here with numpy, in place, into the caller's float32 buffer.
Only the SDK runner module is loaded: the package imports OpenCV and PyAudio
for its camera and microphone helpers, which the server never uses.
The .eim stderr is logged; the last lines also explain why a load failed.
"""

import collections
import importlib.util
import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from multiprocessing import shared_memory

import numpy as np

START_TIMEOUT_S = 10  # .eim start, including NPU context creation
REQUEST_TIMEOUT_S = 10  # one request to the .eim (hello or inference)
STOP_TIMEOUT_S = 3  # grace after SIGINT before SIGKILL
STDERR_LINES = 20  # lines of .eim output kept for diagnostics

log = logging.getLogger("ei.runner")
F32 = np.float32


def _sdk_runner_module():
    """The SDK runner module alone, without the package __init__ and its OpenCV and PyAudio imports."""
    package = importlib.util.find_spec("edge_impulse_linux")
    path = os.path.join(package.submodule_search_locations[0], "runner.py")
    spec = importlib.util.spec_from_file_location("edge_impulse_linux.runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ImpulseRunner = _sdk_runner_module().ImpulseRunner


class RunnerExited(RuntimeError):
    """The .eim process is gone: it closed its socket or exited."""


class Runner(ImpulseRunner):
    def __init__(self, path: str):
        super().__init__(path)
        self._name = os.path.basename(path)
        self._rx = bytearray()  # reply bytes received so far
        self._stderr = collections.deque(maxlen=STDERR_LINES)
        self._stderr_lock = threading.Lock()
        self._stderr_thread = None
        self.grayscale = False

    # ------------------------------------------------------------ lifecycle
    def init(self, debug: bool = False) -> dict:
        """Start the .eim and return its description (SDK "hello" reply)."""
        if not os.access(self._model_path, os.X_OK):
            raise RuntimeError(f"{self._model_path} is missing or not executable")
        self._tempdir = tempfile.mkdtemp()
        socket_path = os.path.join(self._tempdir, "runner.sock")
        self._runner = subprocess.Popen([self._model_path, socket_path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(self._runner.stderr,), daemon=True)
        self._stderr_thread.start()
        deadline = time.monotonic() + START_TIMEOUT_S
        while not os.path.exists(socket_path):
            code = self._runner.poll()
            if code is not None:
                raise RuntimeError(f"runner exited with code {code}{self.output()}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"runner did not start within {START_TIMEOUT_S} s{self.output()}")
            time.sleep(0.05)
        self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._client.settimeout(REQUEST_TIMEOUT_S)
        self._client.connect(socket_path)
        info = self._hello_resp = self.hello()
        if self._allow_shm:
            self._attach_shared_memory(info)
        params = info["model_parameters"]
        if not params["image_input_width"] or not params["image_input_height"]:
            raise RuntimeError("not an image model")
        self.grayscale = params.get("image_channel_count") == 1
        return info

    def stop(self) -> None:
        """Close the connection and terminate the process (SIGINT, then SIGKILL)."""
        process, self._runner = self._runner, None  # the SDK must not signal it again
        super().stop()  # socket, temp dir, shared memory
        if process is None:
            return
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _attach_shared_memory(self, info: dict) -> None:
        """Same as the SDK init(): map the feature and output buffers it announces."""
        if "features_shm" in info:
            shm_info = info["features_shm"]
            shm = shared_memory.SharedMemory(name=shm_info["name"].lstrip("/"))
            self._input_shm = {
                "shm": shm,
                "type": shm_info["type"],
                "elements": shm_info["elements"],
                "array": np.ndarray((shm_info["elements"],), dtype=np.float32, buffer=shm.buf),
            }
        for shm_info in info.get("freeform_output_shm", []):
            shm = shared_memory.SharedMemory(name=shm_info["name"].lstrip("/"))
            self._freeform_output_shm.append({
                "index": shm_info["index"],
                "shm": shm,
                "type": shm_info["type"],
                "elements": shm_info["elements"],
                "array": np.ndarray((shm_info["elements"],), dtype=np.float32, buffer=shm.buf),
            })

    @property
    def transport(self) -> str:
        """How features reach the .eim: 'shm' (shared memory) or 'json' (one integer per pixel, slow)."""
        return "shm" if self._input_shm is not None else "json"

    # ------------------------------------------------------------ inference
    def features(self, image: np.ndarray, out: np.ndarray, tmp: np.ndarray | None = None, bgr: bool = False) -> np.ndarray:
        """Encode an HxWx3 uint8 image at the model size, RGB or BGR (`bgr`), into `out`, a float32 buffer
        with one value per pixel, as the .eim expects: (r << 16) | (g << 8) | b, or the gray level in all
        three bytes. Values stay below 2**24, so they are exact in float32. The work is done in place; gray
        needs a second buffer like `out`, `tmp`, allocated if not given."""
        flat = image.reshape(-1, 3)
        r, g, b = (flat[:, 2], flat[:, 1], flat[:, 0]) if bgr else (flat[:, 0], flat[:, 1], flat[:, 2])
        if self.grayscale:
            # Luma as in the Edge Impulse SDKs, round(0.299 R + 0.587 G + 0.114 B); the bias rounds exact ties up
            if tmp is None:
                tmp = np.empty_like(out)
            np.multiply(g, F32(0.587), out=tmp)
            np.multiply(r, F32(0.299), out=out)
            out += tmp
            np.multiply(b, F32(0.114), out=tmp)
            out += tmp
            out += F32(0.5005)
            np.floor(out, out=out)
            out *= F32(0x010101)
        else:
            np.copyto(out, r, casting="unsafe")
            out *= F32(256)
            out += g
            out *= F32(256)
            out += b
        return out

    def classify(self, features: np.ndarray) -> dict:
        if self._input_shm is None:
            features = features.astype(np.uint32).tolist()  # sent as JSON
        return super().classify(features)

    def send_msg(self, msg: dict) -> dict:
        """One request to the .eim; replies are JSON objects ended by a NUL byte.
        A reply to an earlier request that timed out is skipped, so one timeout
        does not leave every following request one reply behind."""
        self._ix += 1
        msg["id"] = self._ix
        self._client.sendall(json.dumps(msg).encode())
        try:
            reply = self._read_reply()
            while reply.get("id") != self._ix:
                log.warning("[%s] skipped late reply to request %s", self._name, reply.get("id"))
                reply = self._read_reply()
        except TimeoutError:
            raise RuntimeError(f"no reply from the runner within {REQUEST_TIMEOUT_S} s") from None
        if not reply.get("success"):
            raise RuntimeError(reply.get("error") or "runner reported an error")
        del reply["id"], reply["success"]
        return reply

    def _read_reply(self) -> dict:
        while True:
            end = self._rx.find(b"\0")
            if end >= 0:
                data = bytes(self._rx[:end])
                del self._rx[: end + 1]
                return json.loads(data)
            chunk = self._client.recv(65536)
            if not chunk:
                code = self._exit_code()
                raise RunnerExited(f"runner exited with code {code}" if code is not None else "runner closed its connection")
            self._rx += chunk

    def _exit_code(self):
        """Exit code of the .eim, None if still running; waits briefly since the socket closes before the exit."""
        try:
            return self._runner.wait(1) if self._runner is not None else None
        except subprocess.TimeoutExpired:
            return None

    # ------------------------------------------------------------ diagnostics
    def output(self) -> str:
        """Last stderr lines of the .eim, for load error messages ('' if none).
        Complete once the process has exited: the pipe closes and the drain thread ends."""
        if self._runner is not None and self._runner.poll() is not None:
            self._stderr_thread.join(STOP_TIMEOUT_S)
        with self._stderr_lock:
            lines = list(self._stderr)
        return f"; runner output: {' | '.join(lines)}" if lines else ""

    def _read_stderr(self, pipe) -> None:
        with pipe:
            for line in pipe:
                text = line.decode(errors="replace").strip()
                if text:
                    log.info("[%s] %s", self._name, text)
                    with self._stderr_lock:
                        self._stderr.append(text[:200])
