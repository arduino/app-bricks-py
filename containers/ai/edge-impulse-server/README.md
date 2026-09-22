# edge-impulse-server

Multi-model Edge Impulse inference server, the code shared by two images: this directory is not an
image. [edge-impulse-runner](../edge-impulse-runner/README.md) runs the `.eim` models on the CPU,
[edge-impulse-npu-runner](../edge-impulse-npu-runner/README.md) on the Hexagon NPU through QNN. Their
Dockerfiles copy `src/`, the locked packages and the out-of-the-box models from here through the
`server` build context that `docker-bake.hcl` links to both targets; each image README covers what it adds.

Many clients connect to the same Unix socket. One connection uses one model, requested with the
first message; connections asking for the same model share it, and it is terminated when the last
one closes unless it is pinned. Clients send their frames as they are, the server resizes them to the
model input the way the Studio does and returns the boxes in the frame coordinates, never images. The client library lives in the app-bricks-py library,
outside the containers.

## Models

The models are the executable `.eim` files under `/models`, the model name is their path relative to it
without the extension, subdirectories included. The images bundle the out-of-the-box models under
`/models/ootb/ei` (`ootb/ei/yolo-x-nano`), the CPU builds from `models/ei-ootb-models` here plus, on the
NPU image, its QNN builds; the service mounts the models the app CLI installs next to them,
`edge-impulse/<dir>/<name>` and `custom-ei/<id>/model`.

## Server arguments

| Argument | Required | Meaning |
|---|---|---|
| `--max-models N` | yes | models in memory at the same time, pinned ones included |
| `--max-clients N` | yes | concurrent connections |
| `--memory-reserve-mb N` | yes | minimum available memory required to load a model |
| `--max-model-instances N` | no | instances of one model the server may run when the connections keep it saturated, within the cores and the memory (default 1) |
| `--pinned-models A,B` | no | comma-separated models loaded at startup and never terminated |
| `--models-dir` | no | directory of the `.eim` files (default `/models`) |
| `--socket` | no | socket path (default `/ipc/ei.sock`) |
| `--stats-every` | no | seconds between log summaries (default 60) |
| `--log-level` | no | Python log level (default `WARNING`: only problems; `INFO` adds the model lifecycle, the instances and the periodic stats) |
| `--accel cpu\|qnn` | no | startup checks only (NPU device access); default from `EI_ACCEL`, set by each image |

The compose file of each image passes them in `command`. Startup fails if a pinned model does not
exist or does not load, or if there are more pinned models than `--max-models`.

## Model lifecycle

- Models load one at a time and are terminated only when their last connection closes: inferences,
  frame errors and elapsed time never terminate one.
- The same model runs one inference at a time, different models run in parallel.
- A `.eim` not ready within 10 s fails with `load_failed`, an inference with no reply within 10 s
  fails with `internal`, a `.eim` that dies while loaded is restarted in place. Everything a `.eim`
  writes on stderr is logged with its name.
- The load log line says whether features reach the `.eim` through shared memory (`shm`) or as JSON.
  JSON costs several ms per frame: rebuild such a `.eim` with a recent Edge Impulse release.

## Thresholds and object tracking

A `.eim` exposes its threshold blocks, `thresholds` in `OPND`: each has an `id` and a `type`, the object
detection block its `min_score`, the object tracking block its `max_age`, `min_hits` and `iou_threshold`
(`threshold`, a distance in pixels, for the centroid models such as FOMO). A connection changes them with
`CONF {"id": block, key: value, ...}` and receives the blocks as they stand, or `bad_request` for an
unknown block or key. The values are set on every instance of the model, on the ones added later too, and
belong to the model: they hold for every connection using it.

A model with the object tracking block reports `object_tracking` true in `OPND` and its `tracks` in every
result, the boxes with the `id` of the object, stable while it stays in view. Such a model runs on one
instance whatever the ceiling, see below.

## Model instances

A `.eim` process runs one inference at a time, and the Edge Impulse builds are single-threaded, so on a
board with idle cores a model that its clients keep saturated can go faster with more processes. With
`--max-model-instances` above 1 the server watches the utilization of every model, the busy time of its
instances over the elapsed time, and adds an instance when it stays above 90% for a few seconds, as long
as the ceiling is not reached, the cores allow it (the affinity mask and the cgroup quota against the
cores the `.eim` was measured to use during its warm-up) and the memory reserve holds after another
process of the size of the first. An instance whose model stays under 40% busy for a while is retired.
A client slower than the model never triggers an instance, so the NPU image keeps the default of 1 and
the CPU compose file for the UNO Q sets 2. An instance whose `.eim` exits is replaced on its own, the
others keep serving.

Two guards keep the instances honest. A model that keeps state from one frame to the next, object tracking
in the first place, stays on one instance whatever the ceiling: two trackers would each see half the frames
and number the objects differently. The server reads it from the parameters the `.eim` reports and logs
`single instance` with the reason. And the cores an instance uses are not assumed but measured, at warm-up
and then continuously from the CPU time of the process while it is busy, so a multi-threaded `.eim` that
already fills the cores gets no replica, and instances that turn out to use more cores than the container
has together lose one.

Instances are the server's business: a connection learns only how many frames it may keep in flight,
its slots, and the server picks the moment to raise them so the results stay evenly spaced (half an
inference period after a result). When the results drift closer than two thirds of their target spacing,
the allowance goes back to one until the moment that puts the next frame a full spacing after the
surviving result, so the offset is restored in one step at the cost of a pause as long as the drift.
Results go out in arrival order; one completing after a later frame was answered is dropped, which also
covers an instance that never answers. The stats log reports the instances of each model.

## Unix socket interface

Unix socket `SOCK_STREAM`. Every message: type (4 bytes) + length (uint32 LE) + payload.

```
client: OPEN {"model": name}          first message, mandatory
server: OPND {details}                once the model is ready
        ERR  {...} and close          if it cannot be opened
client: FRAM  ->  server: RSLT | ERR  repeated, up to "slots" frames in flight; an ERR here does not close the connection
          server: SLOT                when the frames the client may keep in flight change
client: CONF  ->  server: CONF | ERR  optional, threshold values of the model
```

| Type | Direction | Payload |
|---|---|---|
| `OPEN` | C -> S | JSON `{"model": name}` |
| `OPND` | S -> C | JSON: `model`, `project`, `width`, `height`, `channels`, `labels`, `model_type`, `resize_mode`, `object_tracking`, `thresholds`, `slots` |
| `FRAM` | C -> S | `<QqHHBB2x` (seq, ts_ns, width, height, channels, color 0=RGB 1=BGR) + the pixels, any size up to 1920x1080 |
| `RSLT` | S -> C | JSON: `seq`, `ts_ns`, `boxes` [{label, score, x, y, w, h}] and `tracks` [{label, score, x, y, w, h, id}] in frame coordinates, `classes`, `anomaly`, `timing_ms`, `slots` |
| `SLOT` | S -> C | JSON: `slots`, the frames the client may keep in flight from now on |
| `CONF` | C -> S | JSON: `id` of a threshold block and the values to set, `{"id": 28, "max_age": 3}` |
| `CONF` | S -> C | JSON: `thresholds`, the blocks with their current values, once the values are set |
| `ERR ` | S -> C | JSON: `op` (`open`, `frame` or `configure`), `code`, `error`, optional `model`/`seq`, `slots` after open |

`slots` is 1 at open and follows the instances of the model; a client that ignores it and sends one frame
at a time keeps working.

| Code | When | Connection |
|---|---|---|
| `bad_request` | first message is not `OPEN`, unexpected message afterwards, nothing sent for 10 s after connecting, or a `CONF` naming an unknown block or key | closed / open |
| `too_many_clients` | `--max-clients` reached | closed |
| `unknown_model` | `<name>.eim` is missing | closed |
| `too_many_models` | `--max-models` reached (the message lists the models in memory) | closed |
| `low_memory` | available memory below `--memory-reserve-mb` | closed |
| `load_failed` | the `.eim` does not start | closed |
| `bad_frame` | invalid frame size or format (a larger frame is discarded, not buffered) | open |
| `internal` | error during inference | open |

The `RSLT` boxes are in the coordinates of the submitted frame: the server keeps the resize transform of
each frame and maps the model output back through it.

| Key | Time spent |
|---|---|
| `recv` | receiving the frame payload once its header arrived (socket transfer) |
| `resize` | adapting the frame to the model input, in the model's resize mode |
| `encode` | packing the pixels into the feature array |
| `lock` | waiting for a free instance of the model, busy with another connection's frame |
| `inference` | the request to the `.eim`: feature copy, its processing, the reply |
| `dsp`, `nn` | reported by the `.eim`, included in `inference`; `inference - dsp - nn` is its own overhead |
| `server` | `resize + encode + lock + inference`, from complete frame to result |

With `--stats-every` and `--log-level INFO` the server logs the frames processed, their rate and the mean of each value
per loaded model, so a running server can be tuned from its log alone.

## Tests and benchmarks

The suite runs the real server against fake `.eim` processes that exit at start, hang, reply late,
die or return errors, covering every protocol path, limit and error code. It runs once for both
images, in the `.venv` of this directory, and needs no board:

```bash
task init:containers    # once, creates the .venv with the test dependency group
task test:containers
```

The fake `.eim` files are executed from a temporary directory, so where `/tmp` is mounted `noexec`
point `EI_TEST_DIR` at a directory that allows execution. `EI_TEST_TIMEOUT` raises the server
timeouts on a slow machine. The benchmark under `tests/` measures a running server on a board:
frame rate, round-trip latency and the `timing_ms` breakdown.
