# edge-impulse-runner

Multi-model Edge Impulse inference server, CPU variant: the `.eim` models run on the CPU. The NPU
variant is [edge-impulse-npu-runner](../edge-impulse-npu-runner/README.md), the same server on
`qairt-common-base`.

Many clients connect to the same Unix socket. One connection uses one model, requested with the
first message; connections asking for the same model share it, and it is terminated when the last
one closes unless it is pinned. Clients send frames already at the model input size, the server
never resizes and never returns images. The client library lives in the app-bricks-py library,
outside the containers.

## Models

The models are the executable `<name>.eim` files in the directory mounted at `/models`; the model
name is the file name. This image runs models built for the *Linux (AARCH64)* deployment target,
models built for the QNN target belong to the NPU image.

## Server arguments

| Argument | Required | Meaning |
|---|---|---|
| `--max-models N` | yes | models in memory at the same time, pinned ones included |
| `--max-clients N` | yes | concurrent connections |
| `--memory-reserve-mb N` | yes | minimum available memory required to load a model |
| `--pinned-models A,B` | no | comma-separated models loaded at startup and never terminated |
| `--models-dir` | no | directory of the `.eim` files (default `/models`) |
| `--socket` | no | socket path (default `/ipc/ei.sock`) |
| `--stats-every` | no | seconds between log summaries (default 60) |
| `--log-level` | no | Python log level (default `INFO`) |
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

## Unix socket interface

Unix socket `SOCK_STREAM`. Every message: type (4 bytes) + length (uint32 LE) + payload.

```
client: OPEN {"model": name}          first message, mandatory
server: OPND {details}                once the model is ready
        ERR  {...} and close          if it cannot be opened
client: FRAM  ->  server: RSLT | ERR  repeated; an ERR here does not close the connection
```

| Type | Direction | Payload |
|---|---|---|
| `OPEN` | C -> S | JSON `{"model": name}` |
| `OPND` | S -> C | JSON: `model`, `project`, `width`, `height`, `channels`, `labels`, `model_type`, `resize_mode` |
| `FRAM` | C -> S | `<QqHHB3x` (seq, ts_ns, width, height, channels) + RGB24 pixels at the model resolution |
| `RSLT` | S -> C | JSON: `seq`, `ts_ns`, `boxes` [{label, score, x, y, w, h}], `classes`, `anomaly`, `timing_ms` |
| `ERR ` | S -> C | JSON: `op` (`open` or `frame`), `code`, `error`, optional `model`/`seq` |

| Code | When | Connection |
|---|---|---|
| `bad_request` | first message is not `OPEN`, unexpected message afterwards, or nothing sent for 10 s after connecting | closed / open |
| `too_many_clients` | `--max-clients` reached | closed |
| `unknown_model` | `<name>.eim` is missing | closed |
| `too_many_models` | `--max-models` reached (the message lists the models in memory) | closed |
| `low_memory` | available memory below `--memory-reserve-mb` | closed |
| `load_failed` | the `.eim` does not start | closed |
| `bad_frame` | invalid frame size or format (a larger frame is discarded, not buffered) | open |
| `internal` | error during inference | open |

The `RSLT` boxes are in model pixels; the client converts them to the coordinates of the submitted
image. `timing_ms` breaks down that frame in the server:

| Key | Time spent |
|---|---|
| `recv` | receiving the frame payload once its header arrived (socket transfer) |
| `encode` | packing the pixels into the feature array |
| `lock` | waiting for the model, busy with another connection's frame |
| `inference` | the request to the `.eim`: feature copy, its processing, the reply |
| `dsp`, `nn` | reported by the `.eim`, included in `inference`; `inference - dsp - nn` is its own overhead |
| `server` | `encode + lock + inference`, from complete frame to result |

With `--stats-every` the server logs the frames processed, their rate and the mean of each value
per loaded model, so a running server can be tuned from its log alone.

## Tests and benchmarks

The suite runs the real server against fake `.eim` processes that exit at start, hang, reply late,
die or return errors, covering every protocol path, limit and error code. It needs no board:

```bash
task init:containers    # once, creates the .venv with the test dependency group
task test:containers
```

The fake `.eim` files are executed from a temporary directory, so where `/tmp` is mounted `noexec`
point `EI_TEST_DIR` at a directory that allows execution. `EI_TEST_TIMEOUT` raises the server
timeouts on a slow machine. The benchmark under `tests/` measures a running server on a board:
frame rate, round-trip latency and the `timing_ms` breakdown.
