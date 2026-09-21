# edge-impulse-npu-runner

Multi-model Edge Impulse inference server, NPU image: the `.eim` models run on the Hexagon NPU through QNN. The server is [edge-impulse-server](../edge-impulse-server/README.md), which documents the arguments, the model lifecycle, the protocol and the tests; the CPU image is [edge-impulse-runner](../edge-impulse-runner/README.md). This page covers what the NPU image adds.

## Image

Build it by hand:

```bash
docker build --build-context server=../edge-impulse-server -t edge-impulse-npu-runner .
```
