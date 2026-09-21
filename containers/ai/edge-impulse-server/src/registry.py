# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model registry with reference counting.

The available models are the <name>.eim files in the models directory, in subdirectories too: the
name of /models/custom-ei/abc/model.eim is custom-ei/abc/model.

  - pinned (--pinned-models): loaded at startup, never terminated;
  - all the others: loaded when the first connection opens them, shared
    between connections, terminated when the last connection closes.
    Nothing else terminates a model (inferences, errors, elapsed time).

Before loading a model:
  1. the models in memory must be fewer than --max-models;
  2. the available memory must be at least --memory-reserve-mb.
Models are loaded one at a time, so each check sees the real state with no
half-finished concurrent loads. Opening an already loaded model does not wait.

States:  unloaded -> loading -> loaded -> unloading -> unloaded
"""

import collections
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import memory
from preprocess import BGR, RGB, Preprocessor
from runner import Runner, RunnerExited

log = logging.getLogger("ei.registry")

UNLOADED, LOADING, LOADED, UNLOADING = "unloaded", "loading", "loaded", "unloading"
# One path segment of a model name, names may nest as <dir>/<name> under the models directory
MODEL_NAME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MB = 1024 * 1024


class RegistryError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RunnerDied(RegistryError):
    """The .eim of a loaded model exited: the model must be restarted (ModelRegistry.restart)."""

    def __init__(self, message: str):
        super().__init__("internal", message)


# ====================================================================== model
class Model:
    """A running .eim file (child process started by the SDK)."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.runner = Runner(str(path))
        self._lock = threading.Lock()  # one inference at a time per model
        self._closed = False
        self.dead: str | None = None  # why the .eim is gone, once it is
        self._stats = collections.Counter()  # sums of the frame timings since the last profile()
        self._stats_lock = threading.Lock()
        self._stats_since = time.perf_counter()
        try:
            info = self.runner.init()
            params = info["model_parameters"]
            self.width = params["image_input_width"]
            self.height = params["image_input_height"]
            self.labels = list(params.get("labels", []))
            self.model_type = params.get("model_type", "classification")
            self.resize_mode = params.get("image_resize_mode", "squash")
            self.preprocess = Preprocessor(self.width, self.height, self.resize_mode)
            self.project = info["project"]["name"]
            self.infer(np.zeros((self.height, self.width, 3), np.uint8))  # warm-up
        except BaseException:
            self.runner.stop()  # never leave a process behind after a failed load
            raise

    def describe(self) -> dict:
        return {
            "model": self.name,
            "project": self.project,
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "labels": self.labels,
            "model_type": self.model_type,
            "resize_mode": self.resize_mode,
        }

    def scratch(self) -> np.ndarray:
        """Per-connection work buffer for infer(): the encoded features and a temporary of the same size."""
        return np.empty(2 * self.width * self.height, np.float32)

    def infer(self, frame: np.ndarray, color: int = RGB, recv_ms: float = 0.0, scratch: np.ndarray | None = None) -> dict:
        """Run one inference on a frame of any size, resized to the model input. An error here does not terminate the model.
        `color` is the channel order of the frame (protocol RGB or BGR), `recv_ms` the frame transfer time measured
        by the caller, reported with the other timings, `scratch` the caller's buffer from scratch(), allocated here if None."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RegistryError("bad_frame", f"image of shape {frame.shape}, expected HxWx3")
        if scratch is None:
            scratch = self.scratch()
        pixels = self.width * self.height
        features, tmp = scratch[:pixels], scratch[pixels:]
        t0 = time.perf_counter()
        # Resizing and encoding run outside the lock: only the copy to the .eim and the request are serialized.
        image, transform = self.preprocess(frame)
        t_resized = time.perf_counter()
        self.runner.features(image, features, tmp, bgr=color == BGR)
        t1 = time.perf_counter()
        with self._lock:
            t2 = time.perf_counter()
            if self.dead:
                raise RunnerDied(self.dead)
            if self._closed:
                raise RegistryError("internal", "model terminated")
            try:
                raw = self.runner.classify(features)
            except RunnerExited as exc:
                self.dead = str(exc)
                raise RunnerDied(self.dead) from exc
            t3 = time.perf_counter()

        result = raw.get("result", {})
        timing = raw.get("timing", {})
        ms = {
            "recv": recv_ms,
            "resize": (t_resized - t0) * 1e3,
            "encode": (t1 - t_resized) * 1e3,
            "lock": (t2 - t1) * 1e3,
            "inference": (t3 - t2) * 1e3,
            "dsp": timing.get("dsp", 0),
            "nn": timing.get("classification", 0),
            "server": (t3 - t0) * 1e3,
        }
        with self._stats_lock:
            self._stats.update(ms)
            self._stats["frames"] += 1
        boxes = []
        for b in result.get("bounding_boxes") or []:
            x, y, w, h = transform.to_source(b["x"], b["y"], b["width"], b["height"])
            boxes.append({"label": b["label"], "score": float(b["value"]), "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)})
        return {
            "boxes": boxes,
            "classes": result.get("classification") or {},
            "anomaly": float(result.get("anomaly") or 0.0),
            "timing_ms": {key: round(value, 2) for key, value in ms.items()},
        }

    def profile(self) -> dict | None:
        """Frames, rate and mean timings since the previous call; None if no frame was processed."""
        with self._stats_lock:
            stats, self._stats = self._stats, collections.Counter()
            since, self._stats_since = self._stats_since, time.perf_counter()
        frames = stats.pop("frames", 0)
        if not frames:
            return None
        return {"frames": frames, "fps": frames / (self._stats_since - since), "mean_ms": {key: value / frames for key, value in stats.items()}}

    def close(self) -> None:
        """Terminate the .eim process, after any inference in progress."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.runner.stop()


# ====================================================================== registry
@dataclass
class Entry:
    name: str
    path: Path
    pinned: bool = False
    state: str = UNLOADED
    model: Model | None = None
    users: int = 0  # connections that have the model open


class ModelRegistry:
    def __init__(self, models_dir: str, pinned: list, max_models: int, memory_reserve_mb: int):
        self.models_dir = Path(models_dir)
        self.pinned = list(pinned)
        self.max_models = max_models
        self.memory_reserve = memory_reserve_mb * MB
        self._entries: dict[str, Entry] = {}
        self._cond = threading.Condition()
        self._load_lock = threading.Lock()  # one load at a time
        if len(self.pinned) > max_models:
            raise ValueError(f"{len(self.pinned)} models in --pinned-models but --max-models is {max_models}")

    # ------------------------------------------------------------ startup and shutdown
    def start(self) -> None:
        """Load the pinned models; an error here prevents the server from starting."""
        for name in self.pinned:
            entry = self._entry(name)
            with self._cond:
                entry.pinned, entry.state = True, LOADING
            self._load(entry)

    def shutdown(self) -> None:
        with self._cond:
            models = [e.model for e in self._entries.values() if e.model is not None]
        for model in models:
            model.close()

    # ------------------------------------------------------------ connections
    def acquire(self, name: str) -> Model:
        """Register a connection on the model and return it, loading it if needed.
        Blocks until the model is ready or loading fails (RegistryError)."""
        entry = self._entry(name)
        with self._cond:
            entry.users += 1
            while entry.state in (LOADING, UNLOADING):
                self._cond.wait()  # another connection is loading/terminating it
            if entry.state == LOADED:
                return entry.model
            entry.state = LOADING  # under the same lock: only this connection loads it
        try:
            self._load(entry)
            return entry.model
        except BaseException:
            self.release(name)
            raise

    def restart(self, model: Model) -> Model:
        """Replace a model whose .eim exited (RunnerDied) and return the model now in its slot.
        Connections hitting the same exit wait for the first restart. Fails with load_failed."""
        entry = self._entries[model.name]
        with self._cond:
            while entry.state in (LOADING, UNLOADING):
                self._cond.wait()
            if entry.state == LOADED and entry.model is not model:
                return entry.model  # already replaced by another connection
            entry.state, entry.model = LOADING, None
        model.close()  # the dead process and its buffers
        log.warning("[%s] %s: restarting it", model.name, model.dead)
        self._load(entry)
        return entry.model

    def release(self, name: str) -> None:
        """Call when a connection closes. If it was the last one and the model
        is not pinned, the model is terminated."""
        with self._cond:
            entry = self._entries[name]
            entry.users -= 1
            if entry.users > 0 or entry.pinned or entry.state != LOADED:
                return
            entry.state = UNLOADING
            model = entry.model
        model.close()
        with self._cond:
            entry.model, entry.state = None, UNLOADED
            self._cond.notify_all()
        log.info("[%s] terminated: no connection is using it", name)

    def status(self) -> list:
        """Models in memory: (name, connections, pinned)."""
        with self._cond:
            return [(e.name, e.users, e.pinned) for e in self._entries.values() if e.state != UNLOADED]

    def loaded(self) -> list:
        with self._cond:
            return [e.model for e in self._entries.values() if e.state == LOADED]

    # ------------------------------------------------------------ details
    def _entry(self, name: str) -> Entry:
        path = self.models_dir.joinpath(*f"{name}.eim".split("/"))
        if not all(MODEL_NAME_SEGMENT.match(part) for part in str(name).split("/")) or not path.is_file():
            raise RegistryError("unknown_model", f"model '{name}' not found: {self.models_dir}/{name}.eim is missing")
        with self._cond:
            return self._entries.setdefault(name, Entry(name, path))

    def _load(self, entry: Entry) -> None:
        """Load the model; entry.state is already LOADING."""
        try:
            with self._load_lock:
                self._check_limits(entry)
                start = time.perf_counter()
                model = Model(entry.name, entry.path)
                elapsed = time.perf_counter() - start
        except Exception as exc:
            with self._cond:
                entry.state = UNLOADED
                self._cond.notify_all()
            if isinstance(exc, RegistryError) and not isinstance(exc, RunnerDied):
                raise  # a .eim dying during the warm-up is a load failure
            log.error("[%s] loading failed: %s", entry.name, exc)
            raise RegistryError("load_failed", f"model '{entry.name}' failed to start: {exc}") from exc

        with self._cond:
            entry.model, entry.state = model, LOADED
            self._cond.notify_all()
        log.info(
            "[%s] loaded in %.1f s: input %dx%d, %s, resize '%s', features via %s%s",
            entry.name,
            elapsed,
            model.width,
            model.height,
            model.model_type,
            model.resize_mode,
            model.runner.transport,
            " (pinned)" if entry.pinned else "",
        )
        if model.runner.transport != "shm":
            log.warning(
                "[%s] the .eim has no shared memory input: features travel as JSON, "
                "expect several ms more per frame; rebuild it with a recent Edge Impulse release",
                entry.name,
            )

    def _check_limits(self, entry: Entry) -> None:
        with self._cond:
            while True:
                in_memory = [e.name for e in self._entries.values() if e.state in (LOADED, UNLOADING)]
                terminating = any(e.state == UNLOADING for e in self._entries.values())
                if len(in_memory) < self.max_models or not terminating:
                    break
                self._cond.wait()  # a slot is being freed
        if len(in_memory) >= self.max_models:
            raise RegistryError(
                "too_many_models",
                f"cannot load '{entry.name}': {len(in_memory)} models already "
                f"in memory ({', '.join(sorted(in_memory))}), limit --max-models "
                f"{self.max_models}",
            )
        available = memory.available_bytes()
        if available is None or available < self.memory_reserve:
            shown = "unreadable" if available is None else f"{available // MB} MB"
            raise RegistryError(
                "low_memory",
                f"cannot load '{entry.name}': available memory {shown}, required minimum --memory-reserve-mb {self.memory_reserve // MB} MB",
            )
