#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-model Edge Impulse inference server (protocol v8).

Many clients, one socket. Each connection requests a model with its first
message and waits until it is ready. Connections that request the same
model share it; the model is terminated when the last one closes, unless
it is pinned.

  python3 inference_server.py --models-dir /models --pinned-models detector \\
      --max-models 3 --max-clients 8 --memory-reserve-mb 512

Acceleration depends on the .eim files, not on this code:
  - QCS8275 -> deployment "... (AARCH64 with Qualcomm QNN)"  -> NPU
  - QRB2210 -> deployment "Linux (AARCH64)"                 -> CPU
"""

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import memory
import protocol as P
from registry import ModelRegistry, RegistryError, RunnerDied

log = logging.getLogger("ei")

LOGGED_ERRORS = 20  # errors logged at WARNING per connection
OPEN_TIMEOUT_S = 10  # a connection must send its OPEN within this time


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
        "--accel", default=os.environ.get("EI_ACCEL", "cpu"), choices=["cpu", "qnn"], help="only used for startup checks (default from EI_ACCEL)"
    )
    ap.add_argument("--stats-every", type=float, default=60, help="seconds between log summaries")
    ap.add_argument("--log-level", default="INFO")
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
    The model is released when the connection closes."""

    def __init__(self, sock: socket.socket, registry: ModelRegistry, name: str):
        self.sock, self.registry, self.name = sock, registry, name
        self.reader = P.Reader(sock, limit=P.MAX_CONTROL_PAYLOAD)  # raised to the frame size once open
        self.model = None
        self.scratch = None  # work buffer for the model, sized on open
        self.errors = 0

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
        log.info("[%s] requests '%s'", self.name, name)
        try:
            self.model = self.registry.acquire(name)  # blocks until loaded
        except RegistryError as exc:
            self.error("open", exc.code, str(exc), model=name)
            return False
        self.reader.limit = P.frame_size(self.model.width, self.model.height)
        self.scratch = self.model.scratch()
        P.send_json(self.sock, P.OPENED, self.model.describe())
        return True

    def serve_frames(self) -> None:
        # Inference runs here, in the connection thread: different connections
        # work in parallel, the same model processes one frame at a time.
        while True:
            try:
                kind, payload = self.reader.read()
            except P.PayloadTooLarge as exc:
                if exc.kind == P.FRAME:
                    self.error(
                        "frame",
                        P.E_BAD_FRAME,
                        f"frame of {exc.length} bytes, expected {self.reader.limit} for {self.model.width}x{self.model.height} RGB",
                        seq=P.frame_seq(exc.head),
                    )
                else:
                    self.error("frame", P.E_BAD_REQUEST, f"expected FRAM, received {exc.kind!r}")
                continue
            if kind != P.FRAME:
                self.error("frame", P.E_BAD_REQUEST, f"expected FRAM, received {kind!r}")
                continue
            seq = P.frame_seq(payload)
            try:
                seq, ts_ns, image = P.parse_frame(payload)
                result = self.model.infer(image, self.reader.transfer_ms, self.scratch)
            except P.ProtocolError as exc:
                self.error("frame", P.E_BAD_FRAME, str(exc), seq=seq)
            except RunnerDied as exc:
                self.error("frame", exc.code, str(exc), seq=seq)
                self.restart_model()
            except RegistryError as exc:
                self.error("frame", exc.code, str(exc), seq=seq)
            except Exception as exc:
                log.debug("[%s] inference error on '%s'", self.name, self.model.name, exc_info=True)
                self.error("frame", P.E_INTERNAL, str(exc), seq=seq)
            else:
                P.send_json(self.sock, P.RESULT, {"seq": seq, "ts_ns": ts_ns, **result})

    def restart_model(self) -> None:
        """The .eim exited: get the restarted model, or keep the dead one so the next frame retries."""
        try:
            self.model = self.registry.restart(self.model)
        except RegistryError:
            pass  # already logged by the registry

    def close(self) -> None:
        self.sock.close()
        if self.model is not None:
            log.info("[%s] closed, releasing '%s'", self.name, self.model.name)
            self.registry.release(self.model.name)
            self.model = None

    def error(self, op: str, code: str, message: str, **extra) -> None:
        # A misbehaving client must not fill the log: after LOGGED_ERRORS its errors are logged at DEBUG
        self.errors += 1
        level = logging.WARNING if self.errors <= LOGGED_ERRORS else logging.DEBUG
        log.log(level, "[%s] %s: %s", self.name, code, message)
        if self.errors == LOGGED_ERRORS:
            log.warning("[%s] further errors of this connection are logged at DEBUG", self.name)
        P.send_json(self.sock, P.ERROR, {"op": op, "code": code, "error": message, **extra})


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
            "Listening on %s | max %d models, %d connections, reserve %d MB",
            path,
            self.args.max_models,
            self.args.max_clients,
            self.args.memory_reserve_mb,
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
            models = [f"{n} ({u} conn{', pinned' if p else ''})" for n, u, p in self.registry.status()]
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
                    "[%s] %d frames, %.1f fps | mean ms: recv %.1f, encode %.1f, lock %.1f, "
                    "inference %.1f (dsp %.1f, nn %.1f, other %.1f), server %.1f",
                    model.name,
                    profile["frames"],
                    profile["fps"],
                    ms["recv"],
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
        registry = ModelRegistry(args.models_dir, args.pinned_models, args.max_models, args.memory_reserve_mb)
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
