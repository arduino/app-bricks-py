# edge-impulse-npu-runner

Multi-model Edge Impulse inference server, NPU image: the `.eim` models run on the Hexagon NPU through QNN. The server lives in [edge-impulse-runner](../edge-impulse-runner/README.md), the CPU image, which documents the arguments, the model lifecycle, the protocol and the tests. This page covers what the NPU image adds.

## Image

Build it by hand:

```bash
docker build --build-context server=../edge-impulse-runner -t edge-impulse-npu-runner .
```
