# edge-impulse-npu-runner

Multi-model Edge Impulse inference server, NPU variant: the `.eim` models run on the Hexagon NPU
through QNN. The server is the one of [edge-impulse-runner](../edge-impulse-runner/README.md), which
documents the arguments, the model lifecycle, the protocol and the tests. This page covers what the
NPU image changes.

## Models

The image bundles the out-of-the-box models under `/models/ootb/ei`, the QNN builds included
(`ootb/ei/yolo-x-nano-qnn`); the service mounts the models the app CLI installs next to them. The `.eim` files must be built for the *Qualcomm Dragonwing IQ 8275 EVK (AARCH64 with Qualcomm QNN)*
deployment target, int8 quantized. A `.eim` built for the CPU target loads too, but runs on the CPU.
If a QNN model fails to load with QNN errors in the log ("Failed to load skel", "Transport layer
setup failed"), the FastRPC side is not reachable: check the NPU devices and the host DSP mount
the compose file passes in. The `.eim` then falls back to the CPU and the `nn` timings look like the
CPU image ones.

## Tests

The same suite as the CPU container, run with the fake `.eim` processes in CPU mode:

```bash
task init:containers    # once, creates the .venv with the test dependency group
task test:containers
```
