# edge-impulse-npu-runner

Multi-model Edge Impulse inference server, NPU image: the `.eim` models run on the Hexagon NPU through QNN. The server lives in [edge-impulse-runner](../edge-impulse-runner/README.md), the CPU image, which documents the arguments, the model lifecycle, the protocol and the tests. This page covers what the NPU image adds.

## Models

The QNN builds of the out-of-the-box models go in `models/ei-ootb-models/arm64/` here, and only this image
copies them, next to the CPU builds it takes from edge-impulse-runner. A brick that needs a QNN model adds
it to this folder; the first one also adds the `COPY` line, since a `COPY` matching no file fails the build.

## Image

Build it by hand:

```bash
docker build --build-context server=../edge-impulse-runner -t edge-impulse-npu-runner .
```
