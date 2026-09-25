# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Compile the EasyOCR graphs for the Hexagon NPU and write the HTP context binaries.

Why
---
The first QNN session on a model compiles its graph for the HTP. Graph finalization is
single-threaded inside QNN and takes minutes on the recognizer (130 s measured on the 21q
board with finalization mode 0, 6.5 min with mode 3; the detector takes ~2 s). ONNX
Runtime can serialize the compiled graph as an
"EP context" model - `<model>.soc<soc_id>.qnn_ctx.onnx` - that later sessions load in
0.25 s instead of compiling. This script produces those files so they can be committed
next to the models and shipped in the image: no container ever pays the compile.

The name carries the SoC id (/sys/devices/soc0/soc_id, e.g. `recognizer.soc675.qnn_ctx.onnx`
for a QCS8275), so one image can ship one binary per supported SoC and the runner picks
the one matching the board. Run this script once per SoC to support and commit all of them.

A context binary is valid only for the exact combination that produced it:

  * the SoC / HTP architecture of the board it was compiled on
  * the QAIRT release, i.e. the `onnxruntime-qnn` wheel pinned in requirements.txt
  * the model file
  * the compile-time provider options in `utils.onnx_ep.DEFAULT_QNN_OPTIONS`

That combination is recorded next to each binary as `<model>.soc<soc_id>.qnn_ctx.json` and
checked by the runner at start-up (see `utils/onnx_ep.py`); a binary whose recorded soc_id
differs from the board's is refused. Commit the `.json` together with the `.onnx`. Rebuild
both whenever any of the four items above changes.

Where to run it
---------------
On the target board, with the NPU reachable. Either directly:

    uv venv .venv --python 3.13 && source .venv/bin/activate
    uv pip install --require-hashes -r ../aihub-onnx-models-runner/requirements.txt
    python tools/compile_htp_context.py

or inside the runner image, so the QAIRT is exactly the one that will run in production
(devices and mounts are the ones from the brick's compose file):

    docker run --rm -it \\
        --device /dev/dma_heap/system --device /dev/fastrpc-cdsp \\
        -v /usr/share/qcom:/run/host-qcom:ro \\
        -v /sys/devices/soc0:/sys/devices/soc0:ro \\
        -v "$PWD/models:/app/models" \\
        --entrypoint /aihub-onnx-entrypoint.sh \\
        ghcr.io/arduino/app-bricks/ocr-runner:<tag> \\
        python tools/compile_htp_context.py

Either way the binaries land in models/easyocr-onnx-float/ (or --output-dir), next to the
graphs they were compiled from.

What it does
------------
For each graph it opens a QNN session exactly the way the runner does (same provider
options, CPU fallback allowed), which compiles the graph and writes the context binary and
its fingerprint. The node placement is then measured from an ORT profile - not inferred
from `get_providers()`, which lists QNN even when it executed nothing - and the script
fails if the NPU ran no nodes. Finally it reopens each model from the binary just written
and reports the warm start time, which is what a container will pay.

A few CPU nodes are normal: on the recognizer QNN leaves 4 `DequantizeLinear` nodes on the
CPU (~2 % of the time). Many partitions are not normal - every cut round-trips tensors
between NPU and CPU.

Usage
-----
    python tools/compile_htp_context.py                 # both graphs, into the model directory
    python tools/compile_htp_context.py --force         # recompile even if binaries exist
    python tools/compile_htp_context.py --model recognizer --output-dir /tmp/ctx

Environment: EASYOCR_QNN_FINALIZATION_MODE, EASYOCR_QNN_SOC_MODEL, EASYOCR_QNN_HTP_ARCH,
EASYOCR_QNN_VTCM_MB and the other EASYOCR_QNN_* variables documented in utils/onnx_ep.py
are honoured, and the compile-time ones end up in the fingerprint. Whatever you set here
must also be set in production, or the runner will report a fingerprint mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RUNNER_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the EasyOCR graphs for the Hexagon NPU and write the HTP context binaries.",
        epilog="See the module docstring for where to run this and what the output is valid for.",
    )
    parser.add_argument(
        "--model",
        choices=("detector", "recognizer"),
        action="append",
        help="Compile only this graph (repeatable). Default: both.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the .qnn_ctx.onnx/.json files. Default: the model directory, where the runner looks for shipped binaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing binaries first. Without it an existing, matching binary is loaded and reported instead of recompiled.",
    )
    parser.add_argument(
        "--skip-reload-check",
        action="store_true",
        help="Do not reopen each model from the written binary to measure the warm start.",
    )
    return parser.parse_args()


def configure_environment(output_dir: str | None) -> None:
    """Environment the runner modules read at import/session time. Must run before importing them."""
    os.environ["EASYOCR_EP"] = "qnn"  # fail instead of silently compiling nothing on the CPU
    os.environ["EASYOCR_QNN_CONTEXT_CACHE"] = "1"
    os.environ["EASYOCR_QNN_CONTEXT_STRICT"] = "1"  # a binary from another setup is recompiled, not reused
    if output_dir:
        os.environ["EASYOCR_QNN_CONTEXT_DIR"] = os.path.abspath(output_dir)


def report_environment() -> None:
    import onnxruntime as ort

    from utils.onnx_ep import _plugin_module, _plugin_qnn_version, _qnn_provider_options

    plugin = _plugin_module()
    options = _qnn_provider_options(plugin)
    print(f"onnxruntime       {ort.__version__}")
    print(f"onnxruntime-qnn   {getattr(plugin, '__version__', 'NOT INSTALLED')}  (QAIRT {_plugin_qnn_version(plugin)})")
    print(f"HTP backend       {options['backend_path']}")
    compile_options = {k: v for k, v in options.items() if k not in ("backend_path", "htp_performance_mode", "profiling_level")}
    print(f"compile options   {compile_options}")
    if plugin is None:
        print("\nThe QNN plugin EP is missing: pip install --require-hashes -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)


def print_placement(placement: dict[str, tuple[int, float]]) -> None:
    from utils.onnx_ep import QNN_EP_NAME

    total_us = sum(duration for _, duration in placement.values()) or 1.0
    for provider, (count, duration) in sorted(placement.items(), key=lambda item: -item[1][1]):
        unit = "partitions" if provider == QNN_EP_NAME else "nodes"
        print(f"    {provider:<24} {count:>4} {unit:<11} {duration / 1000:8.2f} ms  {duration / total_us * 100:5.1f}%")
    cpu_nodes = sum(count for provider, (count, _) in placement.items() if provider != QNN_EP_NAME)
    partitions = placement.get(QNN_EP_NAME, (0, 0.0))[0]
    if cpu_nodes:
        print(f"    -> {partitions} NPU partition(s), {cpu_nodes} node(s) on the CPU. Fine while the CPU share stays small.")
    else:
        print("    -> whole graph on the NPU.")


def compile_model(label: str, model_path: str, force: bool, reload_check: bool) -> tuple[str, str]:
    """Compile one graph; returns the (binary, fingerprint) paths written."""
    from utils.model_io_processing import ONNXModel
    from utils.onnx_ep import _context_write_path, _find_context_binary, _fingerprint_path

    if not os.path.isfile(model_path):
        raise SystemExit(f"{label}: {model_path} not found (run from the runner directory, or fix EASYOCR_*_MODEL)")

    target = _context_write_path(model_path)
    existing = _find_context_binary(model_path)
    if existing and force:
        for path in (existing, _fingerprint_path(existing)):
            if os.path.isfile(path):
                os.remove(path)
                print(f"{label}: removed {path}")
        existing = None

    print(f"\n{label}: {'loading existing binary ' + existing if existing else 'compiling ' + model_path} ...", flush=True)
    t_start = time.perf_counter()
    model = ONNXModel(model_path, backend="qnn", verify_placement=True)  # raises if the NPU executed nothing
    elapsed = time.perf_counter() - t_start
    print(f"{label}: {'loaded' if existing else 'compiled'} in {elapsed:.1f} s, session on {model.execution_provider}")
    print_placement(model.placement or {})

    binary = existing or target
    fingerprint = _fingerprint_path(binary)
    if not os.path.isfile(binary):
        raise SystemExit(f"{label}: the session came up but {binary} was not written. See the [ocr-ep] log lines above.")
    print(f"{label}: binary      {binary}  ({os.path.getsize(binary) / 1e6:.1f} MB)")
    if os.path.isfile(fingerprint):
        with open(fingerprint, encoding="utf-8") as handle:
            print(f"{label}: fingerprint {fingerprint}\n    " + json.dumps(json.load(handle), sort_keys=True))
    else:
        print(f"{label}: WARNING no fingerprint written at {fingerprint}; the runner will not be able to validate the binary")

    del model
    if reload_check:
        t_start = time.perf_counter()
        reloaded = ONNXModel(model_path, backend="qnn", verify_placement=True)
        print(f"{label}: warm start from the binary in {time.perf_counter() - t_start:.2f} s ({reloaded.execution_provider})")
        del reloaded

    return binary, fingerprint


def main() -> int:
    args = parse_args()
    configure_environment(args.output_dir)

    from utils.constants import DETECTOR_MODEL_PATH, RECOGNIZER_MODEL_PATH

    os.chdir(RUNNER_DIR)  # constants hold paths relative to the runner directory
    report_environment()

    models = {
        "detector": os.environ.get("EASYOCR_DETECTOR_MODEL", DETECTOR_MODEL_PATH),
        "recognizer": os.environ.get("EASYOCR_RECOGNIZER_MODEL", RECOGNIZER_MODEL_PATH),
    }
    selected = args.model or list(models)

    written: list[str] = []
    for label in selected:
        written.extend(compile_model(label, models[label], args.force, not args.skip_reload_check))

    print("\nDone. Files to commit next to the models (both the .onnx and the .json):")
    for path in written:
        print(f"  {os.path.relpath(path, RUNNER_DIR)}")
    print("They are valid only for this SoC, this onnxruntime-qnn wheel and these compile options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
