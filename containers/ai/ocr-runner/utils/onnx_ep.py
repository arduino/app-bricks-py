# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""ONNX Runtime session factory with optional Qualcomm QNN (Hexagon NPU) acceleration.

This replaces the LiteRT `aihub.tf.load_qnn_delegate()` helper the other runners use.
Where TFLite reaches the NPU through an external delegate .so, ONNX Runtime reaches it
through the QNN execution provider, which ships in two flavours:

  * plugin EP (`onnxruntime-qnn` >= 2.x, maintained by Qualcomm) - a standalone wheel
    registered at runtime against a stock `onnxruntime` install. It is the only variant
    with Linux aarch64 wheels, so it is the one the container installs (the base image,
    aihub-onnx-models-runner, pins it in requirements.in). The wheel ships a full
    QAIRT of its own (libQnnHtp.so, libQnnHtpPrepare.so, libQnnHtpV*Skel.so, ~190 MB) and
    that is the QAIRT the NPU runs: the base image builds on python-slim and deliberately
    ships no QAIRT SDK, since the EP refuses one older than the release it was built
    against anyway, and the EP releases built with older QAIRTs miscompile the recognizer
    (see the README). EASYOCR_QNN_BACKEND_PATH can still point the EP at a system copy in
    an image that has one; `_check_backend_matches_plugin` then verifies at start-up that
    it is the release the wheel was built with.
  * bundled EP (`onnxruntime-qnn` 1.x, Windows arm64/x64) - ships QNN inside the ORT
    wheel itself and shows up directly in `ort.get_available_providers()`.

Both are handled here; the plugin is preferred and the bundled build is the fallback.

Environment variables
---------------------
EASYOCR_EP                  auto (default) | qnn | cpu
EASYOCR_QNN_BACKEND_PATH    explicit QnnHtp.dll / libQnnHtp.so path (default: the wheel's
                            copy if present, else the bare name resolved by the dynamic
                            loader, i.e. a system QAIRT)
EASYOCR_QNN_ADSP_PATH       ADSP_LIBRARY_PATH for the DSP skel libraries (Linux)
EASYOCR_QNN_KEEP_ADSP_PATH  1 - keep an inherited ADSP_LIBRARY_PATH as it is
EASYOCR_QNN_PERF_MODE       sustained_high_performance (default), burst, balanced, ...
EASYOCR_QNN_FINALIZATION_MODE  0 (default, fastest compile) .. 3 (slowest compile)
EASYOCR_QNN_FP16            enable_htp_fp16_precision override (default 1)
EASYOCR_QNN_OFFLOAD_IO_QUANT  offload_graph_io_quantization override (default 1)
EASYOCR_QNN_SOC_MODEL       QNN SoC id, lets the HTP compile for a specific target
EASYOCR_QNN_HTP_ARCH        HTP architecture number (68, 69, 73, 75, 79, ...)
EASYOCR_QNN_VTCM_MB         VTCM budget in MB
EASYOCR_QNN_PROFILING       off (default) | basic | detailed (QNN-internal profiling)
EASYOCR_QNN_OP_TRACE        1 - dump the ONNX-op -> QNN-op mapping
EASYOCR_ORT_PROFILE         1 - write an ORT profile showing the QNN/CPU node split
EASYOCR_QNN_CONTEXT_CACHE   1 (default) - cache the compiled HTP graph next to the model
EASYOCR_QNN_CONTEXT_DIR     where to put that cache (defaults to the model directory)
EASYOCR_QNN_CONTEXT_STRICT  1 - recompile instead of loading a binary built elsewhere
EASYOCR_QNN_STRICT          1 - fail instead of silently running subgraphs on the CPU
EASYOCR_QNN_RPC_LATENCY     per-run RPC control latency in microseconds (e.g. 100)
EASYOCR_ORT_LOG_LEVEL       ORT log severity, 3 = errors only (default); 0 = verbose, prints
                            which nodes QNN actually took
EASYOCR_ORT_THREADS         intra-op thread count
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import threading
from collections import defaultdict

import onnxruntime as ort

# ORT log severity (0 verbose, 1 info, 2 warning, 3 error, 4 fatal) unless EASYOCR_ORT_LOG_LEVEL
# says otherwise. Applied at import too: ORT's device discovery already logs at WARNING
# ("GPU device discovery failed: /sys/class/drm/card0/device/vendor") before any session.
DEFAULT_ORT_LOG_SEVERITY = 3
ort.set_default_logger_severity(int(os.environ.get("EASYOCR_ORT_LOG_LEVEL") or DEFAULT_ORT_LOG_SEVERITY))

QNN_EP_NAME = "QNNExecutionProvider"
CPU_EP_NAME = "CPUExecutionProvider"

# QNN ships the HTP backend as a plain shared library; the EP dlopen()s it by name.
DEFAULT_HTP_LIBRARY = "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"

# The DSP-side libraries the HTP backend loads through FastRPC. Host library and skel
# must come from the same QAIRT release.
SKEL_GLOB = "libQnnHtpV*Skel.so"

# Provider options applied to every QNN session.
#   htp_performance_mode                      clock/DCVS policy while the graph runs.
#                                             sustained_high_performance, not burst: for
#                                             burst ORT also requests RPC polling QoS
#                                             (rpc_polling_time=9999 -> fastrpc
#                                             RPC_POLL_QOS), which the container's fastrpc
#                                             1.0.6 rejects; QNN then drops the whole power
#                                             config, DCVS included, ORT logs "Unable to set
#                                             HTP power configurations" and the HTP runs at
#                                             default clocks: 65 ms per recognizer call
#                                             instead of 15 (measured on QCS8275).
#                                             sustained_high_performance carries no polling
#                                             request and measured identical to burst on the
#                                             host, where fastrpc 1.0.15 accepts polling.
#   htp_graph_finalization_optimization_mode  0 (ORT's default) = compile fastest. Higher
#                                             modes trade startup for runtime, and graph
#                                             finalization is single-threaded and can take
#                                             minutes, so raise it only after measuring
#   enable_htp_fp16_precision                 run whatever float32 survives inside the QDQ
#                                             graph as fp16 on the NPU rather than
#                                             bouncing it back to the CPU
#   offload_graph_io_quantization             keep graph-boundary quantize/dequantize on
#                                             the CPU, so a QDQ graph still takes float I/O
DEFAULT_QNN_OPTIONS = {
    "htp_performance_mode": "sustained_high_performance",
    "htp_graph_finalization_optimization_mode": "0",
    "enable_htp_fp16_precision": "1",
    "offload_graph_io_quantization": "1",
}

# Where Linux exposes the SoC identity. A context binary is HTP code for the architecture
# and VTCM of the SoC it was compiled on, so the SoC is part of its fingerprint and checked
# before loading. Inside the container this directory has to be mounted from the host (the
# brick's compose file does); EASYOCR_SOC_SYSFS points elsewhere for tests.
SOC_SYSFS_DIR = "/sys/devices/soc0"

# Compile-time provider options: a context binary is only valid for the exact values it
# was compiled with, so these are part of its fingerprint.
CONTEXT_COMPILE_OPTIONS = (
    "htp_graph_finalization_optimization_mode",
    "enable_htp_fp16_precision",
    "offload_graph_io_quantization",
    "soc_model",
    "htp_arch",
    "vtcm_mb",
)

_plugin_registered = False
_adsp_checked = False
_setup_lock = threading.RLock()


def _log(message: str) -> None:
    print(f"[ocr-ep] {message}", flush=True)


def _plugin_module():
    """The Qualcomm-maintained plugin EP wheel, or None if it is not installed."""
    try:
        import onnxruntime_qnn
    except ImportError:
        return None
    return onnxruntime_qnn


def _plugin_qnn_version(plugin) -> str:
    """The QAIRT release bundled in the plugin wheel (e.g. '2.49.40'), or 'unknown'."""
    if plugin is None:
        return "none"
    info = getattr(plugin, "build_and_package_info", None)
    return str(getattr(info, "qnn_version", getattr(plugin, "qnn_version", "unknown")))


_backend_versions: dict[str, str] = {}


def _backend_qairt_version(backend_path: str) -> str:
    """
    The QAIRT release a QNN backend library belongs to, read from its `AISW_VERSION: x.y.z`
    build string ('unknown' if the file cannot be read or carries no such string).

    Used to make sure the backend the EP is about to load is the release the wheel was
    built for: the EP only checks the QNN API version, which does not change every release.
    """
    if backend_path in _backend_versions:
        return _backend_versions[backend_path]
    version = "unknown"
    try:
        with open(backend_path, "rb") as handle:
            data = handle.read()
        marker = data.find(b"AISW_VERSION: ")
        if marker >= 0:
            start = marker + len(b"AISW_VERSION: ")
            end = start
            while end < len(data) and data[end : end + 1] in b"0123456789.":
                end += 1
            version = data[start:end].decode() or "unknown"
    except OSError:
        pass
    _backend_versions[backend_path] = version
    return version


def _check_backend_matches_plugin(backend_path: str, plugin) -> None:
    """Warn when the backend library is not the QAIRT release the plugin EP was built with."""
    if plugin is None or not (os.sep in backend_path or "/" in backend_path):
        return  # bundled EP, or a bare name resolved by the loader: nothing to compare against
    backend, expected = _backend_qairt_version(backend_path), _plugin_qnn_version(plugin)
    if backend == "unknown":
        plugin_version = getattr(plugin, "__version__", "?")
        _log(f"warning: could not read the QAIRT version of {backend_path}; expected {expected} (onnxruntime-qnn {plugin_version})")
    elif backend != expected:
        _log(
            f"WARNING: {backend_path} is QAIRT {backend} but onnxruntime-qnn {getattr(plugin, '__version__', '?')} was built with QAIRT "
            f"{expected}. Unset EASYOCR_QNN_BACKEND_PATH to go back to the QAIRT the wheel bundles, or keep the two in lockstep and "
            "recompile the HTP context binaries."
        )


def _register_plugin(plugin) -> bool:
    """Register the plugin EP shared library with this process' ORT instance."""
    with _setup_lock:
        return _register_plugin_locked(plugin)


def _register_plugin_locked(plugin) -> bool:
    global _plugin_registered
    if _plugin_registered:
        return True
    if not hasattr(ort, "register_execution_provider_library"):
        _log(f"onnxruntime {ort.__version__} predates plugin execution providers; upgrade onnxruntime to use onnxruntime-qnn 2.x")
        return False
    try:
        ort.register_execution_provider_library(QNN_EP_NAME, plugin.get_library_path())
    except Exception as exc:  # noqa: BLE001 - any failure here just means "no NPU"
        if "already registered" not in str(exc).lower():
            _log(f"could not register the QNN plugin EP: {exc}")
            return False
    _plugin_registered = True
    return True


def _qnn_provider_options(plugin) -> dict[str, str]:
    options = dict(DEFAULT_QNN_OPTIONS)

    # Backend library: explicit path, else the wheel's own copy, which is what the
    # container has - else the bare name, which the dynamic loader resolves through
    # ld.so.cache, for a host that has a QAIRT installed system-wide.
    backend_path = os.environ.get("EASYOCR_QNN_BACKEND_PATH")
    if not backend_path and plugin is not None and hasattr(plugin, "get_qnn_htp_path"):
        try:
            candidate = plugin.get_qnn_htp_path()
        except Exception:  # noqa: BLE001 - fall back to the bare library name
            candidate = None
        if candidate and os.path.isfile(candidate):
            backend_path = candidate
    options["backend_path"] = backend_path or DEFAULT_HTP_LIBRARY

    options["htp_performance_mode"] = os.environ.get("EASYOCR_QNN_PERF_MODE", options["htp_performance_mode"])
    options["htp_graph_finalization_optimization_mode"] = os.environ.get(
        "EASYOCR_QNN_FINALIZATION_MODE", options["htp_graph_finalization_optimization_mode"]
    )
    options["profiling_level"] = os.environ.get("EASYOCR_QNN_PROFILING", "off")
    # Compile-time options, overridable for experiments; both are part of the context
    # binary fingerprint, so changing them means recompiling.
    for env_name, option_name in (
        ("EASYOCR_QNN_FP16", "enable_htp_fp16_precision"),
        ("EASYOCR_QNN_OFFLOAD_IO_QUANT", "offload_graph_io_quantization"),
    ):
        value = os.environ.get(env_name)
        if value:
            options[option_name] = value

    # Target-specific tuning, only forwarded when actually set: an empty or wrong value
    # here is enough to push the whole graph back onto the CPU.
    for env_name, option_name in (
        ("EASYOCR_QNN_SOC_MODEL", "soc_model"),
        ("EASYOCR_QNN_HTP_ARCH", "htp_arch"),
        ("EASYOCR_QNN_VTCM_MB", "vtcm_mb"),
        # 1 = dump the ONNX-op -> QNN-op mapping, including the ops QNN would not take
        ("EASYOCR_QNN_OP_TRACE", "enable_framework_op_trace"),
    ):
        value = os.environ.get(env_name)
        if value:
            options[option_name] = value

    return options


def _session_options(intra_op_threads: int | None, profile: bool = False) -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    threads = intra_op_threads or int(os.environ.get("EASYOCR_ORT_THREADS", "0"))
    if threads:
        session_options.intra_op_num_threads = threads
    if os.environ.get("EASYOCR_QNN_STRICT", "0") == "1":
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile or os.environ.get("EASYOCR_ORT_PROFILE", "0") == "1":
        # Writes a chrome-trace JSON whose per-node events carry args.provider. That is the
        # only reliable way to know which EP actually executed what: get_providers() lists
        # every registered provider, including one that claimed no nodes at all.
        session_options.enable_profiling = True
        session_options.profile_file_prefix = "easyocr_ort_profile"
    # Errors only by default: at WARNING, ORT prints per session that "some nodes were not
    # assigned to the preferred execution providers" (the 4 DequantizeLinear nodes QNN
    # leaves on the CPU, expected) plus Windows-only feature notices. 0 = verbose, at that
    # level ORT prints the node partitioning, i.e. whether the graph really landed on the
    # NPU ("All nodes placed on [QNNExecutionProvider]") or got split.
    severity = int(os.environ.get("EASYOCR_ORT_LOG_LEVEL") or DEFAULT_ORT_LOG_SEVERITY)
    ort.set_default_logger_severity(severity)
    session_options.log_severity_level = severity
    return session_options


def summarize_profile(profile_path: str, runs: int = 1) -> dict[str, tuple[int, float]]:
    """
    Group an ORT profile's kernel events by the execution provider that ran them.

    Parameters
    ----------
    profile_path
        Chrome-trace JSON written by ORT when profiling is enabled.
    runs
        How many inferences the profile covers; totals are divided by it.

    Returns
    -------
    breakdown : dict
        ``{provider: (count, microseconds)}``. QNN emits one event per fused partition
        rather than one per original node, so its count is a partition count while the
        CPU side is a node count - which is exactly the comparison that matters.
    """
    with open(profile_path, encoding="utf-8") as handle:
        events = json.load(handle)

    breakdown: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for event in events:
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
            continue
        provider = event.get("args", {}).get("provider", "unknown")
        breakdown[provider][0] += 1
        breakdown[provider][1] += event.get("dur", 0)

    return {provider: (count // runs, duration / runs) for provider, (count, duration) in breakdown.items()}


def _has_skels(directory: str) -> bool:
    return bool(directory) and os.path.isdir(directory) and bool(glob.glob(os.path.join(directory, SKEL_GLOB)))


def _prepare_adsp_path(backend_path: str) -> None:
    """
    Point ADSP_LIBRARY_PATH, which decides where the DSP loads its skel libraries, at
    skels that match the HTP backend library about to be loaded.

    Host library and skel must come from the same QAIRT release, or the backend fails to
    start with QNN_DEVICE_ERROR_INVALID_CONFIG - and ORT only warns about an inherited
    ADSP_LIBRARY_PATH in passing. The skels that match are the ones the `onnxruntime-qnn`
    wheel ships next to libQnnHtp.so, so they win unless told otherwise. That matters
    wherever something else exports the variable: a LiteRT image built on qairt-common-base
    sets ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp for its own, older QAIRT, and so do hosts
    with a QAIRT SDK installed.
    """
    global _adsp_checked
    with _setup_lock:
        if _adsp_checked:
            return
        _adsp_checked = True

    override = os.environ.get("EASYOCR_QNN_ADSP_PATH")
    if override:
        _set_adsp_path(override)
        _log(f"ADSP_LIBRARY_PATH set to {override}")
        return

    current = os.environ.get("ADSP_LIBRARY_PATH")
    if os.environ.get("EASYOCR_QNN_KEEP_ADSP_PATH", "0") == "1":
        if current and not any(_has_skels(directory) for directory in _split_path(current)):
            _log(f"warning: ADSP_LIBRARY_PATH={current} contains no {SKEL_GLOB}; keeping it as asked")
        return

    backend_dir = os.path.dirname(os.path.abspath(backend_path)) if os.sep in backend_path or "/" in backend_path else ""
    if _has_skels(backend_dir):
        # The expected case: the skels next to the backend library are the matching
        # ones, so switch silently. EASYOCR_QNN_ADSP_PATH / EASYOCR_QNN_KEEP_ADSP_PATH
        # are the documented overrides.
        if current != backend_dir:
            _set_adsp_path(backend_dir)
        return

    if not current:
        return  # ORT installs its own default, which is usually right

    if any(_has_skels(directory) for directory in _split_path(current)):
        return

    # A value inherited from another project points the DSP at skel libraries from a
    # different QAIRT version, and the HTP backend then dies with
    # QNN_DEVICE_ERROR_INVALID_CONFIG. Dropping it is the verified fix: ORT installs its
    # own working default when the variable is absent. Scoped to this process, and
    # announced rather than silent.
    os.environ.pop("ADSP_LIBRARY_PATH", None)
    _log(
        f"unset ADSP_LIBRARY_PATH ({current}) for this process: it contains no {SKEL_GLOB}, "
        "which makes the HTP backend fail to start. ORT will install its own default. "
        "Override with EASYOCR_QNN_ADSP_PATH, or keep yours with EASYOCR_QNN_KEEP_ADSP_PATH=1."
    )


def _split_path(value: str) -> list[str]:
    separator = ";" if ";" in value else ":"
    return [part for part in value.split(separator) if part]


def _set_adsp_path(directory: str) -> None:
    os.environ["ADSP_LIBRARY_PATH"] = directory
    # Images built on qairt-common-base export both; keep them in agreement.
    if "CDSP_LIBRARY_PATH" in os.environ:
        os.environ["CDSP_LIBRARY_PATH"] = directory


def _writable(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK)


def _context_write_dir(model_path: str, announce: bool = False) -> str:
    """
    Where a freshly compiled HTP context binary is written.

    Next to the model by default. That directory is read-only in most container images,
    so fall back to a cache directory rather than losing the NPU: a failed context write
    otherwise takes the whole QNN session down with it. The fallback is only announced
    when `announce` is set, i.e. when a compile is actually about to write there - the
    same lookup runs at every start to find shipped binaries, where it is not news.
    """
    explicit = os.environ.get("EASYOCR_QNN_CONTEXT_DIR")
    if explicit:
        return explicit

    beside_model = os.path.dirname(os.path.abspath(model_path))
    if _writable(beside_model):
        return beside_model

    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    fallback = os.path.join(base, "easyocr-onnx", "qnn-context")
    if _writable(fallback):
        if announce:
            _log(
                f"{beside_model} is not writable; caching compiled HTP graphs in {fallback} instead. "
                "Mount a volume there (or set EASYOCR_QNN_CONTEXT_DIR) to keep them across container restarts."
            )
        return fallback

    fallback = os.path.join(tempfile.gettempdir(), "easyocr-onnx-qnn-context")
    if announce:
        _log(f"no writable cache directory found; falling back to {fallback} (lost on reboot)")
    return fallback


def _context_binary_names(model_path: str) -> list[str]:
    """
    File names a compiled HTP context binary for `model_path` may have, most specific first.

    Binaries are SoC-specific, so they carry the SoC id in their name:
    `<model>.soc<soc_id>.qnn_ctx.onnx`. That lets one image ship one binary per supported
    SoC side by side, and the runner picks the one matching /sys/devices/soc0/soc_id. The
    unsuffixed `<model>.qnn_ctx.onnx` is the name used when the SoC cannot be identified,
    and is accepted as a fallback (its fingerprint is still checked).
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]
    soc_id, _ = _soc_info()
    names = [f"{stem}.qnn_ctx.onnx"]
    if soc_id != "unknown":
        names.insert(0, f"{stem}.soc{soc_id}.qnn_ctx.onnx")
    return names


def _context_write_path(model_path: str) -> str:
    """Where a freshly compiled HTP context binary for `model_path` is written (SoC-specific name when the SoC is known)."""
    return os.path.join(_context_write_dir(model_path, announce=True), _context_binary_names(model_path)[0])


def _find_context_binary(model_path: str) -> str | None:
    """
    Locate an existing compiled HTP context binary for `model_path`.

    Pre-compiled binaries are shipped next to the model, and in an image that directory is
    read-only - so the lookup does not depend on being able to write there. Search order:

      1. the write directory (EASYOCR_QNN_CONTEXT_DIR, or the fallback cache when the
         model directory is read-only) - a binary recompiled on this machine must shadow a
         shipped one that was rejected and could not be deleted;
      2. the model directory, where shipped binaries live.

    In each directory the SoC-specific name is tried before the unsuffixed one.
    """
    names = _context_binary_names(model_path)
    candidates = [_context_write_dir(model_path), os.path.dirname(os.path.abspath(model_path))]
    seen: set[str] = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        for name in names:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return path
    return None


# Compiled context binaries are routinely built on one machine and copied to another - a
# board to a container image, most often. They are only valid for the exact combination
# below, and a mismatch that still loads is worse than one that fails, so the combination
# is recorded next to the binary and checked on the way back in.
def _soc_info() -> tuple[str, str]:
    """
    (soc_id, machine) of the SoC this process runs on, from sysfs.

    Both are 'unknown' when the files cannot be read: a non-Qualcomm development machine,
    or a container without /sys/devices/soc0 mounted from the host.
    """
    directory = os.environ.get("EASYOCR_SOC_SYSFS", SOC_SYSFS_DIR)
    values: list[str] = []
    for name in ("soc_id", "machine"):
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                values.append(handle.read().strip() or "unknown")
        except OSError:
            values.append("unknown")
    return values[0], values[1]


def _context_fingerprint(model_path: str, options: dict[str, str]) -> dict[str, str]:
    plugin = _plugin_module()
    compile_options = {key: value for key, value in options.items() if key in CONTEXT_COMPILE_OPTIONS}
    try:
        stat = os.stat(model_path)
        model_stamp = f"{stat.st_size}"
    except OSError:
        model_stamp = "?"
    soc_id, soc_machine = _soc_info()

    return {
        "onnxruntime": ort.__version__,
        "onnxruntime_qnn": str(getattr(plugin, "__version__", "none")),
        "qnn_version": _plugin_qnn_version(plugin),
        "backend": os.path.basename(options.get("backend_path", "")),
        "backend_qairt": _backend_qairt_version(options.get("backend_path", "")),
        "model_bytes": model_stamp,
        "options": json.dumps(compile_options, sort_keys=True),
        "soc_id": soc_id,
        "soc_machine": soc_machine,
    }


def _fingerprint_path(cache_path: str) -> str:
    return f"{os.path.splitext(cache_path)[0]}.json"


def _write_fingerprint(cache_path: str, model_path: str, options: dict[str, str]) -> None:
    try:
        with open(_fingerprint_path(cache_path), "w", encoding="utf-8") as handle:
            json.dump(_context_fingerprint(model_path, options), handle, indent=2, sort_keys=True)
    except OSError as exc:
        _log(f"could not record the context binary fingerprint: {exc}")


def _check_fingerprint(cache_path: str, model_path: str, options: dict[str, str]) -> bool:
    """Compare a cached binary against the setup that is about to use it."""
    path = _fingerprint_path(cache_path)
    if not os.path.isfile(path):
        _log(
            f"{os.path.basename(cache_path)} has no fingerprint file next to it, so it cannot be checked "
            "against this setup. If you copied it from another machine, copy the .json alongside it."
        )
        return True

    try:
        with open(path, encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, ValueError) as exc:
        _log(f"unreadable fingerprint {path}: {exc}")
        return True

    current = _context_fingerprint(model_path, options)
    name = os.path.basename(cache_path)

    # The SoC is not negotiable: HTP code compiled for one SoC does not run on another, so
    # a mismatch rejects the binary outright instead of "loading it anyway".
    recorded_soc, current_soc = recorded.get("soc_id"), current["soc_id"]
    if recorded_soc not in (None, "unknown"):
        if current_soc == "unknown":
            _log(
                f"warning: {name} was compiled for {recorded.get('soc_machine')} (soc_id {recorded_soc}) but this SoC cannot be "
                f"identified: {os.environ.get('EASYOCR_SOC_SYSFS', SOC_SYSFS_DIR)}/soc_id is not readable. Mount /sys/devices/soc0 "
                "from the host into the container to enable the check."
            )
        elif current_soc != recorded_soc:
            _log(
                f"ERROR: {name} was compiled for {recorded.get('soc_machine')} (soc_id {recorded_soc}) but this board is "
                f"{current['soc_machine']} (soc_id {current_soc}). HTP context binaries are SoC-specific: not loading it. "
                "Recompile on this board with tools/compile_htp_context.py."
            )
            return False

    # soc_machine is informational (the id is what is compared); everything else recorded
    # must match. Fields a fingerprint predates (written by an older runner) are not compared.
    differences = [
        f"{key}: {recorded[key]!r} -> {current[key]!r}"
        for key in current
        if key in recorded and key != "soc_machine" and not (key == "soc_id" and current_soc == "unknown") and recorded[key] != current[key]
    ]
    if not differences:
        return True

    _log(
        f"WARNING: {name} was compiled under a different setup - "
        + "; ".join(differences)
        + ". Loading it anyway; it is only valid for the SoC, QAIRT version and compile options that produced it. "
        "Delete it to recompile, or set EASYOCR_QNN_CONTEXT_STRICT=1 to recompile automatically on a mismatch."
    )
    return os.environ.get("EASYOCR_QNN_CONTEXT_STRICT", "0") != "1"


def _create_qnn_session(
    model_path: str,
    plugin,
    options: dict[str, str],
    session_options: ort.SessionOptions,
) -> ort.InferenceSession:
    """Create a session on the QNN EP, preferring the plugin registration path."""
    if plugin is not None and _register_plugin(plugin) and hasattr(session_options, "add_provider_for_devices"):
        devices = [device for device in ort.get_ep_devices() if device.ep_name == QNN_EP_NAME]
        if devices:
            session_options.add_provider_for_devices(devices, options)
            return ort.InferenceSession(model_path, sess_options=session_options)
        _log("the QNN plugin EP is installed but reports no compatible device")

    if QNN_EP_NAME in ort.get_available_providers():
        return ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[QNN_EP_NAME],
            provider_options=[options],
        )

    raise RuntimeError("QNNExecutionProvider is not available. On a Snapdragon target install it with `pip install -r requirements.txt`.")


def _try_qnn(model_path: str, intra_op_threads: int | None, profile: bool) -> ort.InferenceSession | None:
    plugin = _plugin_module()
    if plugin is None and QNN_EP_NAME not in ort.get_available_providers():
        _log("QNN EP not installed, using the CPU provider")
        return None

    options = _qnn_provider_options(plugin)
    _check_backend_matches_plugin(options["backend_path"], plugin)
    _prepare_adsp_path(options["backend_path"])
    use_cache = os.environ.get("EASYOCR_QNN_CONTEXT_CACHE", "1") != "0"

    # A compiled HTP context binary skips graph finalization, which otherwise costs
    # minutes of startup on every process launch. The first run writes it, later runs
    # load it instead of the original graph.
    cache_path = _find_context_binary(model_path) if use_cache else None
    if cache_path is not None and not _check_fingerprint(cache_path, model_path, options):
        _log(f"recompiling {os.path.basename(model_path)} because the cached binary does not match")
    elif cache_path is not None:
        try:
            session = _create_qnn_session(cache_path, plugin, options, _session_options(intra_op_threads, profile))
            _log(f"QNN EP attached from cached context binary {cache_path}")
            return session
        except Exception as exc:  # noqa: BLE001 - a stale binary is not fatal, just rebuild
            _log(f"cached context binary rejected ({exc}); recompiling from {model_path}")
            try:
                os.remove(cache_path)
            except OSError:
                _log(f"{cache_path} could not be deleted (read-only?); the recompiled binary will shadow it")

    write_path = _context_write_path(model_path) if use_cache else None
    try:
        session_options = _session_options(intra_op_threads, profile)
        if write_path:
            session_options.add_session_config_entry("ep.context_enable", "1")
            session_options.add_session_config_entry("ep.context_file_path", write_path)
            session_options.add_session_config_entry("ep.context_embed_mode", "1")
        session = _create_qnn_session(model_path, plugin, options, session_options)
    except Exception as exc:  # noqa: BLE001 - unsupported ops, missing libs, wrong SoC, ...
        if not write_path:
            _log(f"QNN EP could not run {os.path.basename(model_path)}: {exc}")
            return None
        # Writing the context binary is an optimisation. Never let it cost us the NPU.
        _log(f"context binary generation failed ({exc}); retrying without it")
        try:
            session = _create_qnn_session(model_path, plugin, options, _session_options(intra_op_threads, profile))
        except Exception as retry_exc:  # noqa: BLE001
            _log(f"QNN EP could not run {os.path.basename(model_path)}: {retry_exc}")
            return None

    if write_path and os.path.isfile(write_path):
        _write_fingerprint(write_path, model_path, options)
        _log(f"compiled HTP context binary written to {write_path}")

    # Deliberately not claiming the NPU here. ORT creates the session even when QNN's
    # GetCapability fails outright and the EP ends up claiming zero nodes; only a profiled
    # run settles it, which is what ONNXModel.verify_placement() does.
    _log(f"QNN EP attached to {os.path.basename(model_path)}, node placement not yet verified")
    return session


def _cpu_session(model_path: str, intra_op_threads: int | None, profile: bool) -> ort.InferenceSession:
    return ort.InferenceSession(
        model_path,
        sess_options=_session_options(intra_op_threads, profile),
        providers=[CPU_EP_NAME],
    )


def build_session(
    model_path: str,
    backend: str | None = None,
    intra_op_threads: int | None = None,
    profile: bool = False,
) -> tuple[ort.InferenceSession, str]:
    """
    Open `model_path` on the best available execution provider.

    Parameters
    ----------
    model_path
        Path to the .onnx graph.
    backend
        "auto" (default) tries QNN and falls back to the CPU, "qnn" requires the NPU,
        "cpu" skips QNN entirely. Defaults to $EASYOCR_EP.
    intra_op_threads
        CPU thread count; 0/None leaves it to ORT.
    profile
        Enable ORT profiling on the session, so the caller can read back which provider
        actually executed which nodes. Stops as soon as `end_profiling()` is called.

    Returns
    -------
    session : ort.InferenceSession
    provider : str
        Name of the provider the session actually landed on.
    """
    backend = (backend or os.environ.get("EASYOCR_EP", "auto")).lower()
    if backend not in ("auto", "qnn", "cpu"):
        raise ValueError(f"Unknown backend {backend!r}, expected auto, qnn or cpu")

    if backend != "cpu":
        session = _try_qnn(model_path, intra_op_threads, profile)
        if session is not None:
            return session, session.get_providers()[0]
        if backend == "qnn":
            raise RuntimeError(
                f"EASYOCR_EP=qnn was requested but {model_path} could not be placed on the "
                "QNN execution provider. See the log lines above for the reason."
            )

    return _cpu_session(model_path, intra_op_threads, profile), CPU_EP_NAME
