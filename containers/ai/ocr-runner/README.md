# ocr-runner: EasyOCR on ONNX Runtime + Qualcomm QNN

Model runner behind the `arduino:ocr` brick. CRAFT text detector + CRNN recognizer from
[Qualcomm AI Hub's EasyOCR](https://aihub.qualcomm.com/models/easyocr) (ai-hub-models
v0.61.0, float export), executed with ONNX Runtime on the Hexagon NPU as fp16 through the
QNN execution provider. Pre/post-processing is numpy/OpenCV only: no torch, no `easyocr`
package.

This used to be a TFLite/LiteRT runner. It was ported because the TFLite recognizer could
not be delegated to the NPU through `libQnnTFLiteDelegate.so` (dynamic-shaped tensor in
the exported graph), so it ran on the CPU at ~250 ms per text box. With ONNX Runtime both
graphs land on the NPU: detector ~85 ms, recognizer ~17 ms per box (QCS8275 / IQ8, float).

## Layout

| path | role |
| --- | --- |
| `inference.py` | the pipeline; `inference_callback` and `apply_config` are what the `aihub` framework calls |
| `utils/onnx_ep.py` | ORT session factory: QNN plugin EP, CPU fallback, HTP context binaries, fingerprints |
| `utils/model_io_processing.py` | `ONNXModel`: NHWC float in/out over the NCHW graphs (and (de)quantization from `metadata.json` for integer exports) |
| `utils/constants.py` | model paths, thresholds, character set |
| `utils/orientation.py` | rotated text: read the whole image turned by 90/180/270 too and keep the orientation that reads most confidently (`rotation` setting) |
| `utils/{bbox,image,post}_processing.py`, `utils/metadata.py` | runtime-agnostic EasyOCR ports (unchanged from the TFLite version) |
| `models/easyocr-onnx-float/` | `.onnx` + `.data` graphs, `metadata.json`, and the compiled `*.soc<id>.qnn_ctx.onnx` / `.json`, one pair per SoC |
| `tools/compile_htp_context.py` | run on the board: compiles both graphs for the HTP and writes the context binaries |

## Client configuration

The brick sends a `{"config": {...}}` message before each frame; `apply_config` in
`inference.py` applies it and unknown keys are ignored:

| key | value | effect |
| --- | --- | --- |
| `allowlist` | string of characters, `""` to clear | only these characters can be decoded (CTC logits of the others are zeroed) |
| `rotation` | list of angles among 90, 180, 270, `[]` to clear | the whole image is also read turned counter-clockwise by each angle, and the orientation with the highest character-weighted mean confidence is reported, a rotated one only when it beats upright by 0.1 (see `utils/orientation.py`). Positions are mapped back to the original frame. One extra full pass, detection and recognition, per angle |

Both settings are process-wide and persist until the next config message, which is why
the brick restates them on every call.

## Accuracy and known limitations

Measured on the CPU (`EASYOCR_EP=cpu`, the float graphs, which read like the HTP ones)
with the runner pipeline on rendered two-line texts: 9 phrases x 7 fonts (Times, Arial,
Courier, Calibri, Georgia, Verdana, Segoe UI) x 2 sizes (48 and 80 px). Letters and digits
read correctly 98.1% of the time. Some punctuation does not:

| character | read correctly | typical error |
| --- | --- | --- |
| `_` | 29% | dropped |
| `/` | 50% | read as `I` |
| `!` | 62% | read as `l` (28 of 84) |
| `.` | 68% | dropped, mostly inside `...` |
| `$` | 79% | read as `s` |
| `,` `:` `@` `#` | 93% | |
| `? ; ( ) + = % " ' & * -` | 100% | |

This is the recognizer, not the pre-processing: padding the cutouts with background
(10-20% of their height) fixes single cases, e.g. `youl` -> `you!` on
`tests/containers/ai/ocr_runner_images/hey-arduino.png`, and breaks as many others
(`Arduino!` -> `Arduinol`); over all the images the word accuracy moves by under 1%.

Rotated text (`rotation`): on the same texts turned 90 degrees clockwise, reading the
whole image turned gets 84% of the words right, as many as upright, and picks the right
orientation for 167 of 168 images. The previous per-box approach (EasyOCR's
`rotation_info`: detect once, recognize each cutout rotated) got 50%. The reasoning is in
the `utils/orientation.py` docstring. Only 90 degrees was measured this way; 180 and 270
use the same code path but have not been benchmarked.

`tests/containers/ai/test_ocr_runner_model.py` reads the two reference images with the
real models; it needs `onnxruntime` and is skipped without it.

## Everything is pinned

Three things decide whether a compiled HTP graph is reusable: the model bytes, the QAIRT
release, and the compile options. Each is fixed in git:

* **Models**: tracked in the repository (`models/easyocr-onnx-float/`), the
  `easyocr-onnx-float.zip` of ai-hub release v0.61.0 unpacked as is.
* **Runtime**: entirely in the base image,
  [`aihub-onnx-models-runner`](../aihub-onnx-models-runner). `onnxruntime`, the
  `onnxruntime-qnn` plugin EP, `numpy` and `opencv-python-headless` are the same for every
  ONNX runner, so they are pinned once there, in `requirements.in`/`.txt` - this runner
  installs nothing of its own. That lock is a `uv pip compile --no-deps --generate-hashes`
  lock for the container target, installed with `pip install --no-deps --require-hashes`, so
  no transitive resolution happens at build time (sympy, mpmath, coloredlogs and
  humanfriendly, ~80 MB the wheels declare but never import, stay out) and a rebuild can
  neither pick a newer `onnxruntime` nor a different `onnxruntime-qnn` wheel nor an extra
  package. The QNN plugin wheel is self-contained: `onnxruntime-qnn 2.5.0` ships
  **QAIRT 2.49.40** (`libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnHtpV68..V81Skel.so`,
  ~190 MB), and it is the *only* QAIRT in the image - the base image builds on `python-slim`
  and installs no QAIRT SDK, precisely because this runner would not use one.
  `utils/onnx_ep.py` points `ADSP_LIBRARY_PATH` at the wheel's own skels: host library and
  DSP skel must come from the same release or the backend fails with
  `QNN_DEVICE_ERROR_INVALID_CONFIG`. That works with the FastRPC libraries the base image
  installs (Debian's `libfastrpc1`, the same quic/fastrpc 1.0.6 `qairt-common-base` builds
  from source): it reads `ADSP_LIBRARY_PATH` with `getenv()` at every file open, for the
  CDSP domain too, and always appends the yaml-derived DSP payload path after it;
  `CDSP_LIBRARY_PATH` is not referenced anywhere in its sources. The only system library the
  backend takes is `libcdsprpc.so`, which is the point - everything else it loads by
  absolute path from its own directory (`dladdr`), build ids checked. To see it on a board:
  `LD_DEBUG=libs python -c "import inference" 2>&1 | grep -E 'QnnHtp|QnnSystem|cdsprpc'`.
* **Why the wheel's QAIRT and not a system one** (measured on the 21q, EasyOCR recognizer,
  one release of `onnxruntime-qnn` at a time, each with its bundled QAIRT):

  | onnxruntime-qnn | onnxruntime | QAIRT | recognizer on the HTP |
  | --- | --- | --- | --- |
  | 2.1.1 | 1.24.4 | 2.45.41 | garbage, every confidence 0.00 |
  | 2.2.0 | 1.24.4 | 2.46.0 | garbage |
  | 2.3.0 | 1.29.0 | 2.47.0 | digits misread, detector no longer on the NPU |
  | 2.4.0 | 1.29.0 | 2.48.40 | correct |
  | 2.5.0 | 1.29.0 | 2.49.40 | correct (pinned) |

  With 2.1.1 the same recognizer reads correctly on the CPU EP, so it is the HTP graph those
  QAIRT releases produce that is wrong, not the model. 2.45.41 is what `qairt-common-base`
  ships, which is one reason this runner no longer builds on it - the other being that a
  QAIRT SDK it cannot use is over a GB of image. Pointing the EP at a system QAIRT with
  `EASYOCR_QNN_BACKEND_PATH` only works if that QAIRT is **at least as new** as the one the
  EP was built against: the EP takes the backend's interface only when the QNN API major
  matches and minor/patch are >= its own. Verified on the 21q, where the host's QAIRT 2.46
  (`qairt-libs`) is refused with `QNN SetupBackend failed Unable to find a valid interface
  for /usr/lib/libQnnHtp.so`. A newer system QAIRT would load, but the context binaries are
  tied to the QAIRT that compiled them and would need recompiling; `utils/onnx_ep.py` warns
  when the backend's QAIRT (`AISW_VERSION` string in the library) is not the plugin's.
* **Compile options**: `DEFAULT_QNN_OPTIONS` in `utils/onnx_ep.py`. The ones that shape the
  binary (`htp_graph_finalization_optimization_mode`, `enable_htp_fp16_precision`,
  `offload_graph_io_quantization`, `soc_model`, `htp_arch`, `vtcm_mb`) are part of its
  fingerprint.

To bump any of them, edit the base image's `requirements.in` and regenerate - the command
is in the file's header - or replace the model files, and then **recompile the context
binaries**.

CI enforces it: `tests/containers/ai/test_ocr_runner_context_binaries.py` compares every
committed `*.qnn_ctx.json` with the `==` pins in the base image's `requirements.txt`,
the `# qairt-version:` line in its `requirements.in`, the compile-time subset of
`DEFAULT_QNN_OPTIONS` and the size of the `.onnx` it was compiled from, and checks they were
compiled on the supported SoC (`soc_id` 675, QCS8275). It also fails if any runner on this
base starts installing an `onnxruntime`/`onnxruntime-qnn`/`numpy`/OpenCV of its own on top
of the base's, which would let the two drift apart. A wheel bump, a model swap or an option change without
recompiled binaries fails the test suite with the recompile command in the message. What CI
cannot check is whether the binaries actually run on a board: that is the runtime `soc_id`
check.

## HTP context binaries (the 7-minute problem)

The first QNN session on a model compiles the graph for the HTP. Graph finalization is
single-threaded inside QNN: on the 21q board (QCS8275, HTP v75: the backend loads `libQnnHtpV75Stub.so`) the detector
takes 3 s and the recognizer 109 s with `htp_graph_finalization_optimization_mode=0`
(minutes more with mode 3). The compiled result is written next to the model as
`<model>.soc<soc_id>.qnn_ctx.onnx` plus a `<model>.soc<soc_id>.qnn_ctx.json` fingerprint,
and every later start loads it in 0.25 s.

The SoC id in the name (`/sys/devices/soc0/soc_id`, 675 for the QCS8275) is what makes a
**multi-SoC image** possible: ship one pair per supported SoC side by side and the runner
loads the one matching the board it is on. Supporting another SoC means running the
compile script once on such a board and committing its pair; `SUPPORTED_SOCS` in
`tests/containers/ai/test_ocr_runner_context_binaries.py` lists the SoCs that must have
binaries. The unsuffixed `<model>.qnn_ctx.onnx` is only written and looked up when the SoC
cannot be identified.

The binaries are compiled once on the target and **committed** (`detector.soc675.qnn_ctx.onnx`
42.3 MB, `recognizer.soc675.qnn_ctx.onnx` 11.2 MB, compiled 2026-09-08 on the 21q board with
`onnxruntime-qnn` 2.5.0 / QAIRT 2.49.40), so no container ever compiles. Measured
placement: both graphs 1 QNN partition, 100 % on the NPU. To regenerate them:

```bash
# on the board, from this directory, with requirements.txt installed
python tools/compile_htp_context.py
```

The script (docstring has the details, including how to run it inside the runner image)
opens each graph on QNN exactly like the runner does, writes the binary and its
fingerprint, measures how much of the graph the NPU really executes and fails if it ran
nothing, then reopens the model from the binary and reports the warm start time. Commit
`models/easyocr-onnx-w8a8/*.soc<id>.qnn_ctx.onnx` and `.json` and rebuild the image.

Lookup order at start-up (`_find_context_binary`):

1. `EASYOCR_QNN_CONTEXT_DIR` if set, otherwise the model directory when writable, otherwise
   `$XDG_CACHE_HOME/easyocr-onnx/qnn-context` (the fallback also used for *writing* when the
   model directory is read-only);
2. the model directory - where the **committed, pre-compiled binaries** live. This works
   with a read-only model directory, so binaries baked into the image are used as-is.

In each directory `<model>.soc<soc_id>.qnn_ctx.onnx` is tried first, then the unsuffixed
name.

A binary is only valid for the SoC/HTP architecture, QAIRT release, model file and compile
options that produced it. The `.json` records `onnxruntime`, `onnxruntime_qnn`,
`qnn_version` (the bundled QAIRT), the backend library, the model size, the compile
options, and the SoC it was compiled on (`soc_id` and `soc_machine` from
`/sys/devices/soc0`, e.g. `675` / `QCS8275`). On load it is compared with the current
setup:

* a **different `soc_id`** rejects the binary outright, with an `ERROR` line naming both
  SoCs: HTP code compiled for one SoC does not run on another. The same SoC on a board from
  another vendor has the same `soc_id`, so it passes. The brick's compose file mounts
  `/sys/devices/soc0` read-only into the container for this check; without it the runner
  logs that the SoC cannot be identified and loads the binary unchecked;
* every other difference is logged by name and the binary is still loaded
  (`EASYOCR_QNN_CONTEXT_STRICT=1` recompiles instead).

A binary QNN itself rejects is deleted and recompiled, and if it cannot be deleted
(read-only image layer) the recompiled one lands in the cache directory and shadows it from
then on.

Without a usable binary the first start of a container pays the compile, well past the
brick's 30 s connection timeout and the compose healthcheck. That is the failure mode a
fingerprint mismatch must never degrade into silently; see the section above for what
invalidates the binaries.

## Running outside Docker (on the board)

```bash
uv venv .venv --python 3.13 && source .venv/bin/activate
uv pip install --require-hashes -r ../aihub-onnx-models-runner/requirements.txt  # aarch64 only; on x86_64 use the .in file (CPU)
python tools/compile_htp_context.py                    # compiles, then reports the measured NPU/CPU split
EASYOCR_QNN_VERIFY=1 python -c "import inference"      # loads both models like the runner and reports the placement
```

Both float graphs run entirely on the NPU; a few CPU nodes would be acceptable, many
partitions are not. If the backend does not come up at all
(`QNN_DEVICE_ERROR_INVALID_CONFIG`, "Failed to create device"), the usual causes in order
are: an `ADSP_LIBRARY_PATH` that does not hold the skels matching `libQnnHtp.so` (ORT's own
warning `Using existing ADSP_LIBRARY_PATH setting of ...` names the directory in use; the
runner switches to the wheel's silently), FastRPC permissions (`/dev/fastrpc-cdsp`
and `/dev/dma_heap/system` not passed to the container), a missing DSP payload (the host's
`/usr/share/qcom` has to be bind-mounted at `/run/host-qcom`, from where
`/aihub-onnx-entrypoint.sh` merges it into `/usr/share/hexagon-dsp`), and an `htp_arch` /
`soc_model` forced through the environment that the SoC does not have. `EASYOCR_ORT_LOG_LEVEL=0` makes
ORT print the QNN error verbatim.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `EASYOCR_EP` | `auto` | `auto` (QNN, CPU fallback) \| `qnn` (fail unless the NPU executes nodes) \| `cpu` |
| `EASYOCR_EP_DETECTOR` / `EASYOCR_EP_RECOGNIZER` | - | per-model override |
| `EASYOCR_DETECTOR_MODEL` / `EASYOCR_RECOGNIZER_MODEL` | `models/easyocr-onnx-w8a8/*.onnx` | model paths |
| `EASYOCR_QNN_CONTEXT_CACHE` | `1` | use/write the compiled HTP graph |
| `EASYOCR_QNN_CONTEXT_DIR` | model directory | where to read/write that cache |
| `EASYOCR_QNN_CONTEXT_STRICT` | `0` | `1` = recompile instead of loading a binary whose fingerprint differs |
| `EASYOCR_QNN_STRICT` | `0` | `1` = refuse to run unless the whole graph is on the NPU |
| `EASYOCR_QNN_VERIFY` | `0` | `1` = measure the real node placement at start-up (implied by `EASYOCR_EP=qnn`) |
| `EASYOCR_QNN_BACKEND_PATH` | wheel's `libQnnHtp.so` if present, else `libQnnHtp.so` via the loader | HTP backend library |
| `EASYOCR_QNN_ADSP_PATH` | wheel directory | `ADSP_LIBRARY_PATH` for the DSP skel libraries |
| `EASYOCR_QNN_KEEP_ADSP_PATH` | `0` | `1` = keep the inherited `ADSP_LIBRARY_PATH` untouched |
| `EASYOCR_QNN_PERF_MODE` | `sustained_high_performance` | `burst`, `balanced`, `power_saver`, ... See the note on `burst` below |
| `EASYOCR_QNN_FINALIZATION_MODE` | `0` | `0` fastest compile ... `3` slowest compile / fastest run (invalidates binaries) |
| `EASYOCR_QNN_FP16` / `EASYOCR_QNN_OFFLOAD_IO_QUANT` | `1` / `1` | `enable_htp_fp16_precision` / `offload_graph_io_quantization` overrides (invalidate binaries) |
| `EASYOCR_QNN_SOC_MODEL` / `EASYOCR_QNN_HTP_ARCH` / `EASYOCR_QNN_VTCM_MB` | - | target-specific tuning (invalidates binaries) |
| `EASYOCR_PARALLEL_INIT` | `0` | `1` = compile the two models on two threads |
| `EASYOCR_QNN_PROFILING` | `off` | `basic` \| `detailed` (QNN-internal profiling) |
| `EASYOCR_QNN_OP_TRACE` | `0` | `1` = dump the ONNX-op to QNN-op mapping |
| `EASYOCR_ORT_PROFILE` | `0` | `1` = write an ORT profile showing the QNN/CPU node split |
| `EASYOCR_QNN_RPC_LATENCY` | - | per-run RPC control latency, microseconds |
| `EASYOCR_ORT_LOG_LEVEL` | `3` | ORT log severity: `3` errors only, `2` warnings, `0` verbose (prints the partitioning) |
| `EASYOCR_ORT_THREADS` | - | intra-op threads for the CPU provider |
| `EASYOCR_DEBUG` | `0` | `1` = stage-by-stage timings and applied config on stdout. One INFO summary line per frame is always logged |

## Why not `burst`

`htp_performance_mode=burst` is the QNN mode with the highest clocks, but with `burst` ORT
also asks fastrpc for RPC polling QoS (`rpc_polling_time=9999`, i.e. `RPC_POLL_QOS`). The
container ships fastrpc 1.0.6 (Debian's `libfastrpc1`, installed by the base image;
`qairt-common-base` builds the same upstream version from source), and there
`manage_poll_qos` fails; QNN rejects the **whole** power configuration, DCVS included, ORT
logs `Unable to set HTP power configurations` and the HTP stays at default clocks. Measured
in the container on the 21q: detector invoke 103 ms and 65 ms per recognizer call with
`burst` or `default`, 22 ms and 15 ms with `sustained_high_performance`, which asks for no
polling. On the host, whose fastrpc is 1.0.15 (`qcom-fastrpc1`), `burst` works and is no
faster than `sustained_high_performance` for this workload. Hence the default. If the base
image moves to a fastrpc that accepts `RPC_POLL_QOS`, `burst` becomes an option again; the
runtime-only nature of this option means switching it never invalidates the context binaries.

## Models

`models/easyocr-onnx-float`: float32 I/O, NCHW, static shapes - detector `[1, 3, 608, 800]`,
recognizer `[1, 1, 64, 800]`. Each graph is `<model>.onnx` plus external weights
`<model>.data`; `metadata.json` describes the I/O (no quantization parameters for this
export; `ONNXModel` reads them from there when an integer export is used instead).
**Keep the three together.**

### Why float and not w8a8

ai-hub also publishes a w8a8 export. It is the same size on disk (a QDQ graph: weights
quantized in value, stored as float32) and it is faster on the HTP, but its HTP execution
degrades the recognizer in a way the same quantized graph on the CPU does not. Measured on
the 21q with `onnxruntime-qnn` 2.5.0, two photos (a printed sheet with 28 text regions and
a barcode label, correct value 7630049202832):

| | barcode | sheet: regions with character errors | typical confidence |
| --- | --- | --- | --- |
| w8a8 on the HTP | 7630049202**88**2 | 4 (`ENERICQ`, `APP TQ`, `SU APP.TQ`, `~SU APPUNTAMENTO`) | 0.5-0.7, one region at 0.19 |
| w8a8 on the CPU | 7630049202**821** | 0 of those | 0.9-1.0 |
| float on the CPU | 7630049202**822** | 0 of those | 0.9-1.0 |
| float on the HTP | 7630049202832 | 1 (`APP To`) | 0.9-1.0 |

Flipping `enable_htp_fp16_precision` or `offload_graph_io_quantization` for the w8a8 graph
changes nothing. The price of float on the HTP:

| | detector invoke | recognizer per box | 1 region | 28 regions | context binaries |
| --- | --- | --- | --- | --- | --- |
| w8a8 | 22 ms | 15 ms | 62 ms | 525 ms | 21.3 + 10.5 MB |
| float | 85 ms | 17 ms | 135 ms | 656 ms | 42.3 + 11.2 MB |

+63 ms per image and +2 ms per text box, for readings that match the exact model and
confidences that stay above the brick's default threshold. The w8a8 export is one env var
away for comparisons (`EASYOCR_DETECTOR_MODEL` / `EASYOCR_RECOGNIZER_MODEL`), its
`metadata.json` carries the scale/zero-point `ONNXModel` needs.
