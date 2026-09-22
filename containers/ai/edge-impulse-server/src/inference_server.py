#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-model Edge Impulse inference server.

Many clients, one socket. Each connection requests a model with its first
message and waits until it is ready. Connections that request the same
model share it; the model is terminated when the last one closes, unless
it is pinned.

  python3 inference_server.py --models-dir /models --pinned-models detector \\
      --max-models 3 --max-clients 8 --memory-reserve-mb 512 --max-model-instances 2

The same server is packaged by two images, acceleration depends on the .eim files and on the image:
  - edge-impulse-npu-runner -> deployment "... (AARCH64 with Qualcomm QNN)" -> NPU
  - edge-impulse-runner     -> deployment "Linux (AARCH64)"                 -> CPU
"""

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import memory
import protocol as P
from registry import ModelRegistry, RegistryError, RunnerDied

log = logging.getLogger("ei")

LOGGED_ERRORS = 20  # errors logged at WARNING per connection
OPEN_TIMEOUT_S = 10  # a connection must send its OPEN within this time
DRIFT_TOLERANCE = 0.35  # results closer than this fraction under their target spacing (period / instances) get re-phased


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=f"/ipc/{P.SOCKET_NAME}", help="socket path")
    ap.add_argument("--models-dir", default="/models", help="directory containing the <name>.eim files")
    ap.add_argument(
        "--pinned-models", type=model_list, default=[], metavar="NAME[,NAME...]", help="comma-separated models loaded at startup and never terminated"
    )
    ap.add_argument("--max-models", type=int, required=True, help="models in memory at the same time, pinned ones included")
    ap.add_argument("--max-clients", type=int, required=True, help="concurrent connections")
    ap.add_argument("--memory-reserve-mb", type=int, required=True, help="minimum available memory required to load a model")
    ap.add_argument(
        "--max-model-instances",
        type=int,
        default=1,
        help="instances of one model the server may run when the connections keep it saturated, within the cores and the memory (default 1)",
    )
    ap.add_argument(
        "--accel", default=os.environ.get("EI_ACCEL", "cpu"), choices=["cpu", "qnn"], help="only used for startup checks (default from EI_ACCEL)"
    )
    ap.add_argument("--stats-every", type=float, default=60, help="seconds between log summaries")
    ap.add_argument("--log-level", default="WARNING", help="WARNING by default: only problems; INFO adds the model lifecycle and the periodic stats")
    return ap.parse_args()


def model_list(text: str) -> list:
    """'a, b,c' -> ['a', 'b', 'c'] (no duplicates, order preserved)."""
    return list(dict.fromkeys(name.strip() for name in text.split(",") if name.strip()))


def check_npu_access(accel: str) -> None:
    """In the NPU container, fail immediately if the hardware is not visible."""
    if accel != "qnn":
        log.info("CPU mode")
        return
    devices = sorted(str(p) for p in Path("/dev").glob("fastrpc-*"))
    if not devices:
        sys.exit("No /dev/fastrpc-*: the NPU is not accessible from the container")
    for dev in devices:
        if not os.access(dev, os.R_OK | os.W_OK):
            sys.exit(f"Missing permissions on {dev} (check group_add)")
    if not os.environ.get("ADSP_LIBRARY_PATH"):
        log.warning("ADSP_LIBRARY_PATH is not set")
    log.info("NPU mode (QNN), devices: %s", devices)


class Connection:
    """A client connection: opens a model, then serves its frames.

    Frames are read, resized and encoded in the connection thread, then run on a free instance of the model
    by a worker, so a connection with several frames in flight keeps several instances busy. Results go out
    in arrival order: one completing after a later frame's result went out is dropped. The connection tells the
    client how many frames it may keep in flight ("slots"): one, or the instances of the model once there are
    several, granted half an inference period after a result so the results stay evenly spaced. When two results
    drift closer than their target spacing, the allowance goes back to one until the moment that puts the next
    frame exactly a half period after the surviving one, so the spacing is restored in one step at the cost of
    a pause as long as the drift. The model is released when the connection closes.
    """

    def __init__(self, sock: socket.socket, registry: ModelRegistry, name: str):
        self.sock, self.registry, self.name = sock, registry, name
        self.reader = P.Reader(sock, limit=P.MAX_CONTROL_PAYLOAD)  # raised to the frame limit once open
        self.model = None
        self.errors = 0
        self.dropped = 0  # results that completed after a later one went out
        self._send_lock = threading.Lock()
        self._workers: ThreadPoolExecutor | None = None
        self._scratches: list = []  # work buffers of the model, one per frame in flight
        self._scratch_lock = threading.Lock()
        self.slots = 1  # frames the client may keep in flight
        self._pipelined = False  # more than one frame was ever allowed in flight: results may complete out of order
        self._last_sent_seq = 0
        self._last_sent_at: float | None = None
        self._grant: threading.Timer | None = None
        self._grant_delay = 0.0  # seconds from the last result to the next grant of the extra slots
        self._closed = False

    def serve(self, accepted: bool) -> None:
        try:
            if self.open_model(accepted):
                self.serve_frames()
        except (ConnectionError, OSError, P.ProtocolError):
            pass  # client disconnected or invalid stream
        except Exception:
            log.exception("[%s] unexpected error", self.name)
        finally:
            self.close()

    def open_model(self, accepted: bool) -> bool:
        """Read the request and reply once the model is ready.
        On error, send ERR; the connection will then be closed."""
        self.sock.settimeout(OPEN_TIMEOUT_S)
        try:
            kind, payload = self.reader.read()
        except P.PayloadTooLarge as exc:
            kind, payload = exc.kind, b""
        except TimeoutError:
            self.error("open", P.E_BAD_REQUEST, f"no OPEN within {OPEN_TIMEOUT_S} s")
            return False
        self.sock.settimeout(None)
        request = P.parse_json(payload) if kind == P.OPEN else None
        name = request.get("model") if request else None
        if not isinstance(name, str):
            self.error("open", P.E_BAD_REQUEST, 'the first message must be OPEN {"model": name}')
            return False
        if not accepted:
            self.error("open", P.E_TOO_MANY_CLIENTS, "connection limit reached (--max-clients)", model=name)
            return False
        log.debug("[%s] requests '%s'", self.name, name)
        try:
            self.model = self.registry.acquire(name)  # blocks until loaded
        except RegistryError as exc:
            self.error("open", exc.code, str(exc), model=name)
            return False
        self.reader.limit = P.MAX_FRAME_PAYLOAD
        self._workers = ThreadPoolExecutor(max_workers=self.registry.max_instances, thread_name_prefix=f"{self.name}-infer")
        self.send(P.OPENED, {**self.model.describe(), "slots": self.slots})
        return True

    def serve_frames(self) -> None:
        while True:
            try:
                kind, payload = self.reader.read()
            except P.PayloadTooLarge as exc:
                if exc.kind == P.FRAME:
                    self.error("frame", P.E_BAD_FRAME, f"frame of {exc.length} bytes, the limit is {P.MAX_FRAME_PAYLOAD}", seq=P.frame_seq(exc.head))
                else:
                    self.error("frame", P.E_BAD_REQUEST, f"expected FRAM, received {exc.kind!r}")
                continue
            if kind != P.FRAME:
                self.error("frame", P.E_BAD_REQUEST, f"expected FRAM, received {kind!r}")
                continue
            seq = P.frame_seq(payload)
            scratch = self._take_scratch()
            try:
                seq, ts_ns, image, color = P.parse_frame(payload)
                prepared = self.model.prepare(image, color, self.reader.transfer_ms, scratch)
            except (P.ProtocolError, RegistryError) as exc:
                self._give_scratch(scratch)
                self.error("frame", P.E_BAD_FRAME, str(exc), seq=seq)
                continue
            # The frame is out of the reader buffer: the inference runs on a worker while the next frame is read
            self._workers.submit(self._infer, seq, ts_ns, prepared, scratch)

    def _infer(self, seq: int, ts_ns: int, prepared, scratch) -> None:
        try:
            result = self.model.infer(prepared)
        except RunnerDied as exc:
            self.error("frame", exc.code, str(exc), seq=seq)
            self.restart_model()
        except RegistryError as exc:
            self.error("frame", exc.code, str(exc), seq=seq)
        except Exception as exc:
            log.debug("[%s] inference error on '%s'", self.name, self.model.name, exc_info=True)
            self.error("frame", P.E_INTERNAL, str(exc), seq=seq)
        else:
            self.send_result(seq, P.RESULT, {"seq": seq, "ts_ns": ts_ns, **result})
            self.registry.autoscale(self.model)
        finally:
            self._give_scratch(scratch)

    def _take_scratch(self):
        with self._scratch_lock:
            return self._scratches.pop() if self._scratches else self.model.scratch()

    def _give_scratch(self, scratch) -> None:
        with self._scratch_lock:
            self._scratches.append(scratch)

    def restart_model(self) -> None:
        """An .eim exited: have its instance replaced, the next frames find the model whole again."""
        try:
            self.registry.restart(self.model)
        except RegistryError:
            pass  # already logged by the registry

    # ------------------------------------------------------------ sending
    def send(self, kind: bytes, obj: dict) -> None:
        with self._send_lock:
            if not self._closed:
                P.send_json(self.sock, kind, obj)

    def send_result(self, seq: int | None, kind: bytes, obj: dict) -> None:
        """Send the result or frame error of `seq` with the current allowance, unless a later frame's already went out."""
        with self._send_lock:
            if self._closed:
                return
            if seq is not None:
                if self._pipelined and seq < self._last_sent_seq:
                    self.dropped += 1
                    log.debug("[%s] result of frame %d dropped, frame %d already answered", self.name, seq, self._last_sent_seq)
                    return
                self._last_sent_seq = seq
            now = time.perf_counter()
            instances = self.model.instance_count
            target = self.model.period / max(1, instances)  # spacing of the results when the instances are interleaved
            if instances < self.slots:
                self.slots = max(1, instances)  # an instance was retired
                self._grant_delay = target
            elif self.slots > 1 and self._last_sent_at is not None and now - self._last_sent_at < (1 - DRIFT_TOLERANCE) * target:
                # The results drifted together: one frame in flight until the moment that puts the next one a full
                # spacing after the result that came before this one, which restores the offset in one step
                self.slots = 1
                self._grant_delay = max(0.0, target - (now - self._last_sent_at))
            elif self.slots < instances:
                self._grant_delay = target  # an instance was added: the extra frame starts a half period after this result
            self._last_sent_at = now
            P.send_json(self.sock, kind, {**obj, "slots": self.slots})
            if self.slots < instances and self._grant is None:
                self._grant = threading.Timer(self._grant_delay, self.grant)
                self._grant.daemon = True
                self._grant.start()

    def grant(self) -> None:
        """Raise the allowance to the instances of the model, at the moment that phases the extra frame between the others."""
        with self._send_lock:
            self._grant = None
            instances = self.model.instance_count
            if self._closed or instances <= self.slots:
                return
            self.slots, self._pipelined = instances, True
            P.send_json(self.sock, P.SLOTS, {"slots": self.slots})

    def error(self, op: str, code: str, message: str, **extra) -> None:
        # A misbehaving client must not fill the log: after LOGGED_ERRORS its errors are logged at DEBUG
        self.errors += 1
        level = logging.WARNING if self.errors <= LOGGED_ERRORS else logging.DEBUG
        log.log(level, "[%s] %s: %s", self.name, code, message)
        if self.errors == LOGGED_ERRORS:
            log.warning("[%s] further errors of this connection are logged at DEBUG", self.name)
        payload = {"op": op, "code": code, "error": message, **extra}
        if op == "frame" and self.model is not None:
            self.send_result(extra.get("seq"), P.ERROR, payload)
        else:
            self.send(P.ERROR, payload)

    def close(self) -> None:
        with self._send_lock:
            self._closed = True
            if self._grant is not None:
                self._grant.cancel()
        if self._workers is not None:
            self._workers.shutdown(wait=True)  # the inferences in flight complete before the model is released
        self.sock.close()
        if self.model is not None:
            log.debug("[%s] closed, releasing '%s'%s", self.name, self.model.name, f", {self.dropped} late results dropped" if self.dropped else "")
            self.registry.release(self.model.name)
            self.model = None


class Server:
    def __init__(self, args, registry: ModelRegistry):
        self.args, self.registry = args, registry
        self.clients = 0
        self.lock = threading.Lock()

    def serve_forever(self) -> None:
        path = Path(self.args.socket)
        path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        mask = os.umask(0o117)  # 0660 from creation, no window with wider access
        try:
            listener.bind(str(path))
        finally:
            os.umask(mask)
        listener.listen()
        log.info(
            "Listening on %s | max %d models, %d connections, reserve %d MB, %d instances per model",
            path,
            self.args.max_models,
            self.args.max_clients,
            self.args.memory_reserve_mb,
            self.args.max_model_instances,
        )
        count = 0
        try:
            while True:
                sock, _ = listener.accept()
                count += 1
                threading.Thread(target=self.handle, args=(sock, f"conn-{count}"), daemon=True).start()
        finally:
            listener.close()
            path.unlink(missing_ok=True)

    def handle(self, sock: socket.socket, name: str) -> None:
        with self.lock:
            accepted = self.clients < self.args.max_clients
            if accepted:
                self.clients += 1
        try:
            Connection(sock, self.registry, name).serve(accepted)
        finally:
            if accepted:
                with self.lock:
                    self.clients -= 1

    def log_stats(self) -> None:
        while True:
            time.sleep(self.args.stats_every)
            models = [f"{n} ({u} conn{', pinned' if p else ''}{f', {i} instances' if i > 1 else ''})" for n, u, p, i in self.registry.status()]
            available = memory.available_bytes()
            log.info(
                "connections %d/%d | models %d/%d: %s | available memory %s MB",
                self.clients,
                self.args.max_clients,
                len(models),
                self.args.max_models,
                ", ".join(models) or "none",
                "?" if available is None else available // (1024 * 1024),
            )
            for model in self.registry.loaded():
                profile = model.profile()
                if profile is None:
                    continue
                ms = profile["mean_ms"]
                log.info(
                    "[%s] %d frames, %.1f fps on %d instance%s | mean ms: recv %.1f, resize %.1f, encode %.1f, lock %.1f, "
                    "inference %.1f (dsp %.1f, nn %.1f, other %.1f), server %.1f",
                    model.name,
                    profile["frames"],
                    profile["fps"],
                    profile["instances"],
                    "" if profile["instances"] == 1 else "s",
                    ms["recv"],
                    ms["resize"],
                    ms["encode"],
                    ms["lock"],
                    ms["inference"],
                    ms["dsp"],
                    ms["nn"],
                    ms["inference"] - ms["dsp"] - ms["nn"],
                    ms["server"],
                )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    check_npu_access(args.accel)
    registry = None
    try:
        registry = ModelRegistry(args.models_dir, args.pinned_models, args.max_models, args.memory_reserve_mb, args.max_model_instances)
        registry.start()
    except (RegistryError, ValueError) as exc:
        if registry is not None:
            registry.shutdown()  # pinned models loaded before the failure
        sys.exit(f"Startup failed: {exc}")
    server = Server(args, registry)

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    threading.Thread(target=server.log_stats, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        registry.shutdown()
        log.info("Stopped")


if __name__ == "__main__":
    main()
