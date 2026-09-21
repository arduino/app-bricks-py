# edge-impulse-runner

Multi-model Edge Impulse inference server, CPU image: the `.eim` models run on the CPU. The server is [edge-impulse-server](../edge-impulse-server/README.md), which documents the arguments, the model lifecycle, the protocol and the tests; the NPU image is [edge-impulse-npu-runner](../edge-impulse-npu-runner/README.md). This page covers what the CPU image adds.

## Image

Build it by hand:

```bash
docker build --build-context server=../edge-impulse-server -t edge-impulse-runner .
```
