# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model registry with reference counting and per-model instances.

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

Instances: a model is one .eim process, serving one inference at a time. When the connections keep it
busy, the registry adds instances of it, up to --max-model-instances and to what the cores and the
memory allow, and retires them when the demand drops: a single-threaded .eim then uses the idle cores.
An instance that dies is replaced on its own, the others keep serving.
"""

import collections
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import memory
from preprocess import BGR, RGB, Preprocessor, Transform
from runner import Runner, RunnerExited, encode_features

log = logging.getLogger("ei.registry")

UNLOADED, LOADING, LOADED, UNLOADING = "unloaded", "loading", "loaded", "unloading"
# One path segment of a model name, names may nest as <dir>/<name> under the models directory
MODEL_NAME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MB = 1024 * 1024

# Scaling policy: utilization is the busy time of the instances over the elapsed time, evaluated every SCALE_INTERVAL
SCALE_INTERVAL = 1.0  # seconds between evaluations
SCALE_UP_UTILIZATION, SCALE_UP_AFTER = 0.9, 3.0  # add an instance when the model stayed this busy for this long
SCALE_DOWN_UTILIZATION, SCALE_DOWN_AFTER = 0.4, 10.0  # retire one when it stayed this idle for this long
MIN_CPU_RATIO = 0.1  # an instance is never counted as cheaper than this many cores
OVERSUBSCRIBED = 1.1  # instances using more than this share of the cores together lose one


def stateful_reason(params: dict) -> str | None:
    """Why a model must stay on one instance: it keeps state from one frame to the next, so two processes would
    each see half the frames and disagree, or its input spans several frames. None for a stateless model."""
    thresholds = [t for t in params.get("thresholds") or [] if isinstance(t, dict)]
    if params.get("has_object_tracking") or any(t.get("type") == "object_tracking" for t in thresholds):
        return "object tracking keeps the tracks between frames"
    if (params.get("image_input_frames") or 1) > 1:
        return f"the input spans {params['image_input_frames']} frames"
    return None


class RegistryError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RunnerDied(RegistryError):
    """The .eim of an instance exited: the model replaces it (Model.replace_dead)."""

    def __init__(self, message: str):
        super().__init__("internal", message)


@dataclass
class Prepared:
    """A frame resized and encoded for the model, ready for any of its instances."""

    features: np.ndarray
    transform: Transform
    ms: dict
    started: float  # perf_counter when the frame was complete


# ====================================================================== instance
class Instance:
    """One running .eim process of a model, serving one inference at a time."""

    def __init__(self, path: Path, ordinal: int):
        self.ordinal = ordinal
        self.runner = Runner(str(path))
        self.busy = False
        self.busy_since = 0.0  # perf_counter of the acquisition, moved forward by each utilization evaluation
        self.draining = False  # being retired: takes no new frame
        self.dead: str | None = None  # why the .eim is gone, once it is
        self._closed = False
        self._lock = threading.Lock()
        self.cpu_ratio = 1.0  # cores the .eim uses while busy: measured at warm-up, then followed while it serves
        self.rss = 0  # bytes of the .eim process after the warm-up
        self._cpu_seen: float | None = None  # CPU seconds of the process at the last utilization evaluation
        self.busy_window = 0.0  # seconds busy since the last evaluation

    @property
    def pid(self) -> int | None:
        process = self.runner._runner
        return process.pid if process is not None else None

    def classify(self, features: np.ndarray) -> dict:
        return self._request(self.runner.classify, features)

    def configure(self, values: dict) -> None:
        """Set threshold values on the .eim, `values` being {"id": block id, key: value, ...} as it takes them."""
        self._request(self.runner.set_threshold, values)

    def _request(self, request, argument):
        """One request to the .eim, which serves one at a time; its exit becomes RunnerDied."""
        with self._lock:
            if self.dead:
                raise RunnerDied(self.dead)
            if self._closed:
                raise RegistryError("internal", "model terminated")
            try:
                return request(argument)
            except RunnerExited as exc:
                self.dead = str(exc)
                raise RunnerDied(self.dead) from exc

    def measure(self, features: np.ndarray) -> dict:
        """The warm-up inference, measuring the cores and the memory the .eim takes."""
        cpu0, wall0 = _cpu_seconds(self.pid), time.perf_counter()
        raw = self.classify(features)
        cpu1, wall1 = _cpu_seconds(self.pid), time.perf_counter()
        if cpu0 is not None and cpu1 is not None and wall1 > wall0:
            self.cpu_ratio = max(MIN_CPU_RATIO, (cpu1 - cpu0) / (wall1 - wall0))
        self._cpu_seen = cpu1
        self.rss = _rss_bytes(self.pid) or 0
        return raw

    def follow_cpu(self, busy: float, elapsed: float) -> None:
        """Refresh the cores the .eim uses from the CPU it consumed while busy since the last call."""
        cpu = _cpu_seconds(self.pid)
        if cpu is not None and self._cpu_seen is not None and busy > 0.2 * elapsed:
            self.cpu_ratio = max(MIN_CPU_RATIO, 0.7 * self.cpu_ratio + 0.3 * (cpu - self._cpu_seen) / busy)
        self._cpu_seen = cpu

    def close(self) -> None:
        """Terminate the .eim process, after any inference in progress."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.runner.stop()


def _cpu_seconds(pid: int | None) -> float | None:
    """User plus system CPU time of a process, from /proc."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def _rss_bytes(pid: int | None) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


# ====================================================================== model
class Model:
    """A loaded .eim file: its parameters, its preprocessing and one or more running instances."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self._cond = threading.Condition()  # instances, their busy flags and the scaling state
        self.instances: list[Instance] = []
        self.scaling = False  # an instance is being added or retired
        self._replacing = False  # dead instances are being replaced: frames wait instead of failing
        self._started = 0  # instances started so far, for their ordinals
        self._stats = collections.Counter()  # sums of the frame timings since the last profile()
        self._stats_lock = threading.Lock()
        self._stats_since = time.perf_counter()
        self._period = 0.0  # seconds of one inference, smoothed
        self._busy_seconds = 0.0  # since the last evaluation
        self._evaluated = time.perf_counter()
        self._high_since: float | None = None
        self._low_since: float | None = None
        self.limited_logged = False  # the refusal to add an instance was logged, the next ones go at DEBUG
        self._overrides: dict = {}  # the threshold values set through configure(), by block id, for the instances added later
        instance = Instance(path, 1)
        try:
            info = instance.runner.init()
            params = info["model_parameters"]
            self.width = params["image_input_width"]
            self.height = params["image_input_height"]
            self.labels = list(params.get("labels", []))
            self.model_type = params.get("model_type", "classification")
            self.resize_mode = params.get("image_resize_mode", "squash")
            self.preprocess = Preprocessor(self.width, self.height, self.resize_mode)
            self.grayscale = instance.runner.grayscale
            self.stateful = stateful_reason(params)  # a reason to stay on one instance, None when replicas are fine
            # The threshold blocks of the .eim with their current values, the ones a connection may change
            self.thresholds = [dict(t) for t in params.get("thresholds") or [] if isinstance(t, dict)]
            self.project = info["project"]["name"]
            self._warm_up(instance)
        except BaseException:
            instance.close()  # never leave a process behind after a failed load
            raise
        self.instances.append(instance)
        self._started = 1

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
            "thresholds": self.current_thresholds(),
        }

    def current_thresholds(self) -> list:
        """The threshold blocks with their current values, copies."""
        with self._cond:
            return [dict(t) for t in self.thresholds]

    @property
    def runner(self) -> Runner:
        """The runner of the first instance, for the load log."""
        return self.instances[0].runner

    @property
    def instance_count(self) -> int:
        with self._cond:
            return len([i for i in self.instances if not i.draining])

    @property
    def period(self) -> float:
        """Seconds one inference takes, smoothed over the recent ones."""
        return self._period

    def scratch(self) -> np.ndarray:
        """Work buffer for prepare(): the encoded features and a temporary of the same size."""
        return np.empty(2 * self.width * self.height, np.float32)

    # ------------------------------------------------------------ inference
    def prepare(self, frame: np.ndarray, color: int = RGB, recv_ms: float = 0.0, scratch: np.ndarray | None = None) -> Prepared:
        """Resize a frame of any size to the model input and encode its features into `scratch`.

        `color` is the channel order of the frame (protocol RGB or BGR), `recv_ms` the frame transfer time
        measured by the caller, reported with the other timings. Runs in the caller's thread, no instance involved.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RegistryError("bad_frame", f"image of shape {frame.shape}, expected HxWx3")
        if scratch is None:
            scratch = self.scratch()
        pixels = self.width * self.height
        features, tmp = scratch[:pixels], scratch[pixels:]
        t0 = time.perf_counter()
        image, transform = self.preprocess(frame)
        t1 = time.perf_counter()
        encode_features(image, features, tmp, bgr=color == BGR, grayscale=self.grayscale)
        t2 = time.perf_counter()
        return Prepared(features, transform, {"recv": recv_ms, "resize": (t1 - t0) * 1e3, "encode": (t2 - t1) * 1e3}, t0)

    def infer(self, prepared: Prepared) -> dict:
        """Run a prepared frame on the first free instance and map the boxes back to the frame coordinates.
        An error here does not terminate the model."""
        t0 = time.perf_counter()
        instance = self._acquire()
        t1 = time.perf_counter()
        try:
            raw = instance.classify(prepared.features)
        finally:
            t2 = time.perf_counter()
            self._release(instance, t2 - t1)
        result = raw.get("result", {})
        timing = raw.get("timing", {})
        ms = {
            **prepared.ms,
            "lock": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "dsp": timing.get("dsp", 0),
            "nn": timing.get("classification", 0),
            "server": (t2 - prepared.started) * 1e3,
        }
        with self._stats_lock:
            self._stats.update(ms)
            self._stats["frames"] += 1
        return {
            "boxes": [self._in_frame(b, prepared.transform) for b in result.get("bounding_boxes") or []],
            "classes": result.get("classification") or {},
            "anomaly": float(result.get("anomaly") or 0.0),
            "timing_ms": {key: round(value, 2) for key, value in ms.items()},
        }

    @staticmethod
    def _in_frame(box: dict, transform: Transform) -> dict:
        """A box of the .eim, in model pixels, mapped to the coordinates of the submitted frame."""
        x, y, w, h = transform.to_source(box["x"], box["y"], box["width"], box["height"])
        return {"label": box["label"], "score": float(box["value"]), "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}

    def configure(self, values: dict) -> list:
        """Set threshold values of one block, `values` being {"id": block id, key: value, ...} as the .eim takes them,
        on every instance and on the ones added later; returns the blocks with their current values. The blocks
        belong to the model: the values hold for every connection using it. bad_request for an unknown block or key."""
        block_id = values.get("id")
        with self._cond:
            block = next((t for t in self.thresholds if t.get("id") == block_id), None)
        if block is None:
            known = ", ".join(f"{t.get('id')} ({t.get('type')})" for t in self.current_thresholds()) or "none"
            raise RegistryError("bad_request", f"no threshold block {block_id!r}, the model has: {known}")
        changes = {key: value for key, value in values.items() if key != "id"}
        knobs = [key for key in block if key not in ("id", "type")]
        unknown = [key for key in changes if key not in knobs]
        if unknown or not changes:
            what = f"not {', '.join(unknown)}" if unknown else "nothing was given"
            raise RegistryError("bad_request", f"block {block_id} ({block.get('type')}) exposes {', '.join(knobs)}, {what}")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in changes.values()):
            raise RegistryError("bad_request", "threshold values must be numbers")
        with self._cond:
            self._overrides.setdefault(block_id, {}).update(changes)
            instances = [i for i in self.instances if not i.dead and not i.draining]
        for instance in instances:
            instance.configure({"id": block_id, **changes})
        with self._cond:
            block.update(changes)
        return self.current_thresholds()

    def _acquire(self) -> Instance:
        """The first free instance, waiting for one; RunnerDied when every instance is gone."""
        with self._cond:
            while True:
                alive = [i for i in self.instances if not i.dead and not i.draining]
                if not alive:
                    if self._replacing:
                        self._cond.wait()
                        continue
                    dead = [i for i in self.instances if i.dead]
                    raise RunnerDied(dead[0].dead if dead else "model terminated")
                for instance in alive:
                    if not instance.busy:
                        instance.busy, instance.busy_since = True, time.perf_counter()
                        return instance
                self._cond.wait()

    def _release(self, instance: Instance, duration: float) -> None:
        with self._cond:
            instance.busy = False
            busy = time.perf_counter() - instance.busy_since
            self._busy_seconds += busy
            instance.busy_window += busy
            self._period = duration if self._period == 0 else 0.9 * self._period + 0.1 * duration
            self._cond.notify_all()

    def _warm_up(self, instance: Instance) -> None:
        prepared = self.prepare(np.zeros((self.height, self.width, 3), np.uint8))
        t0 = time.perf_counter()
        instance.measure(prepared.features)
        with self._cond:
            self._period = self._period or time.perf_counter() - t0

    # ------------------------------------------------------------ instances
    def utilization(self) -> tuple[float, float]:
        """(busy fraction of the instances since the last call, seconds since it), and reset."""
        with self._cond:
            now = time.perf_counter()
            elapsed, self._evaluated = now - self._evaluated, now
            busy, self._busy_seconds = self._busy_seconds, 0.0
            for instance in self.instances:
                if instance.busy:  # the part of the inference in progress that falls in this window
                    busy += now - instance.busy_since
                    instance.busy_window += now - instance.busy_since
                    instance.busy_since = now
                instance.follow_cpu(instance.busy_window, elapsed)
                instance.busy_window = 0.0
            instances = max(1, len([i for i in self.instances if not i.draining]))
        return (min(1.0, busy / (elapsed * instances)) if elapsed > 0 else 0.0), elapsed

    def cpu_used(self) -> float:
        """Cores the instances use together while busy."""
        with self._cond:
            return sum(i.cpu_ratio for i in self.instances)

    def wanted_change(self, cores: float | None) -> tuple[int, str]:
        """(+1, reason) when the instances stayed saturated, (-1, reason) when they stayed idle or use more cores than
        the container has, (0, "") otherwise. Call every SCALE_INTERVAL."""
        utilization, _ = self.utilization()
        now = time.perf_counter()
        if cores is not None and self.instance_count > 1 and self.cpu_used() > OVERSUBSCRIBED * cores:
            return -1, f"the instances use {self.cpu_used():.1f} cores of {cores:.1f}"
        self._high_since = (self._high_since or now) if utilization >= SCALE_UP_UTILIZATION else None
        self._low_since = (self._low_since or now) if utilization <= SCALE_DOWN_UTILIZATION else None
        if self._high_since is not None and now - self._high_since >= SCALE_UP_AFTER:
            self._high_since = now  # the next step needs a full period of saturation again
            return 1, "the model was saturated"
        if self._low_since is not None and now - self._low_since >= SCALE_DOWN_AFTER:
            self._low_since = now
            return -1, "the demand dropped"
        return 0, ""

    def cpu_needed(self) -> float:
        """Cores one more instance would bring the model to."""
        with self._cond:
            ratios = [i.cpu_ratio for i in self.instances]
        return sum(ratios) + max(ratios or [1.0])

    def memory_needed(self) -> int:
        with self._cond:
            return max([i.rss for i in self.instances] or [0])

    def add_instance(self) -> Instance:
        """Start one more instance of the .eim, warmed up, and put it in service."""
        with self._cond:
            self._started += 1
            ordinal = self._started
        instance = Instance(self.path, ordinal)
        try:
            instance.runner.init()
            self._warm_up(instance)
        except BaseException:
            instance.close()
            raise
        with self._cond:
            self.instances.append(instance)
            overrides = [{"id": block_id, **changes} for block_id, changes in self._overrides.items()]
            self._cond.notify_all()
        for values in overrides:  # the thresholds set so far, once in service so none set meanwhile is missed
            try:
                instance.configure(values)
            except RegistryError as exc:
                log.warning("[%s] instance %d: thresholds %s not applied: %s", self.name, instance.ordinal, values, exc)
        return instance

    def retire_instance(self) -> Instance | None:
        """Take the last instance out of service once its inference in progress is done, and terminate it."""
        with self._cond:
            candidates = [i for i in self.instances if not i.draining]
            if len(candidates) < 2:
                return None
            instance = candidates[-1]
            instance.draining = True
            while instance.busy:
                self._cond.wait()
            self.instances.remove(instance)
        instance.close()
        return instance

    def replace_dead(self) -> None:
        """Restart the instances whose .eim exited, one at a time; a failure leaves the model without them."""
        with self._cond:
            dead = [i for i in self.instances if i.dead]
            for instance in dead:
                self.instances.remove(instance)
            self._replacing = bool(dead)
        try:
            for instance in dead:
                instance.close()
                log.warning("[%s] instance %d: %s, restarting it", self.name, instance.ordinal, instance.dead)
                try:
                    replacement = self.add_instance()
                except Exception as exc:
                    log.error("[%s] restarting the instance failed: %s", self.name, exc)
                    raise RegistryError("load_failed", f"model '{self.name}' failed to restart: {exc}") from exc
                log.info("[%s] instance %d replaces %d", self.name, replacement.ordinal, instance.ordinal)
        finally:
            with self._cond:
                self._replacing = False
                self._cond.notify_all()

    # ------------------------------------------------------------ profile and shutdown
    def profile(self) -> dict | None:
        """Frames, rate and mean timings since the previous call; None if no frame was processed."""
        with self._stats_lock:
            stats, self._stats = self._stats, collections.Counter()
            since, self._stats_since = self._stats_since, time.perf_counter()
        frames = stats.pop("frames", 0)
        if not frames:
            return None
        return {
            "frames": frames,
            "fps": frames / (self._stats_since - since),
            "instances": self.instance_count,
            "mean_ms": {key: value / frames for key, value in stats.items()},
        }

    def close(self) -> None:
        """Terminate every .eim process, after the inferences in progress."""
        with self._cond:
            instances, self.instances = self.instances, []
            self._cond.notify_all()
        for instance in instances:
            instance.close()


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
    def __init__(self, models_dir: str, pinned: list, max_models: int, memory_reserve_mb: int, max_instances: int = 1):
        self.models_dir = Path(models_dir)
        self.pinned = list(pinned)
        self.max_models = max_models
        self.memory_reserve = memory_reserve_mb * MB
        self.max_instances = max_instances
        self._entries: dict[str, Entry] = {}
        self._cond = threading.Condition()
        self._load_lock = threading.Lock()  # one load at a time
        self._scale_lock = threading.Lock()
        if len(self.pinned) > max_models:
            raise ValueError(f"{len(self.pinned)} models in --pinned-models but --max-models is {max_models}")
        if max_instances < 1:
            raise ValueError("--max-model-instances must be at least 1")

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
        """Replace the instances of the model whose .eim exited (RunnerDied); connections hitting the same exit wait
        for the first replacement. Fails with load_failed."""
        with self._scale_lock:
            model.replace_dead()
        return model

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
        with self._scale_lock:  # after any instance being added or retired
            model.close()
        with self._cond:
            entry.model, entry.state = None, UNLOADED
            self._cond.notify_all()
        log.info("[%s] terminated: no connection is using it", name)

    def status(self) -> list:
        """Models in memory: (name, connections, pinned, instances)."""
        with self._cond:
            return [(e.name, e.users, e.pinned, e.model.instance_count if e.model else 0) for e in self._entries.values() if e.state != UNLOADED]

    def loaded(self) -> list:
        with self._cond:
            return [e.model for e in self._entries.values() if e.state == LOADED]

    # ------------------------------------------------------------ scaling
    def autoscale(self, model: Model) -> None:
        """Add or retire an instance of the model according to its utilization; call after every inference.
        Evaluates once per SCALE_INTERVAL, the change itself runs in the background."""
        if self.max_instances < 2 or model.stateful:
            return
        with model._cond:
            if model.scaling or time.perf_counter() - model._evaluated < SCALE_INTERVAL:
                return
        change, reason = model.wanted_change(effective_cores())
        if change == 0:
            return
        with model._cond:
            if model.scaling:
                return
            if change > 0 and len(model.instances) >= self.max_instances:
                return
            if change < 0 and len(model.instances) < 2:
                return
            model.scaling = True
        threading.Thread(target=self._scale, args=(model, change, reason), daemon=True, name=f"scale-{model.name}").start()

    def _scale(self, model: Model, change: int, reason: str) -> None:
        try:
            with self._scale_lock:
                if change > 0:
                    self._add_instance(model, reason)
                else:
                    instance = model.retire_instance()
                    if instance is not None:
                        log.info("[%s] instance %d retired: %s, %d left", model.name, instance.ordinal, reason, model.instance_count)
        except Exception as exc:
            log.warning("[%s] instance not added: %s", model.name, exc)
        finally:
            with model._cond:
                model.scaling = False

    def _add_instance(self, model: Model, reason: str) -> None:
        cores = effective_cores()
        limit = None
        if cores is not None and model.cpu_needed() > cores:
            limit = f"{model.cpu_needed():.1f} cores needed, {cores:.1f} available"
        available = memory.available_bytes()
        if limit is None and available is not None and available - model.memory_needed() < self.memory_reserve:
            limit = (
                f"{available // MB} MB available, an instance takes {model.memory_needed() // MB} MB above the {self.memory_reserve // MB} MB reserve"
            )
        if limit is not None:
            log.log(
                logging.DEBUG if model.limited_logged else logging.INFO, "[%s] stays at %d instances: %s", model.name, model.instance_count, limit
            )
            model.limited_logged = True
            return
        start = time.perf_counter()
        instance = model.add_instance()
        model.limited_logged = False
        log.info(
            "[%s] instance %d added in %.1f s: %s, %d instances now",
            model.name,
            instance.ordinal,
            time.perf_counter() - start,
            reason,
            model.instance_count,
        )

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
        if model.stateful and self.max_instances > 1:
            log.info("[%s] single instance: %s", entry.name, model.stateful)

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


def effective_cores() -> float | None:
    """Cores the process may use: its affinity mask, capped by the cgroup CPU quota; None when unknown."""
    try:
        cores = float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return None
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, int(quota) / int(period))
    except (OSError, ValueError):
        pass
    return cores
