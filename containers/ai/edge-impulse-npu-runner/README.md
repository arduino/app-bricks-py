# edge-impulse-npu-runner

Multi-model Edge Impulse inference server, NPU image: the `.eim` models run on the Hexagon NPU through QNN. The server lives in [edge-impulse-server](../../base/edge-impulse-server/README.md), the layer this image adds, which documents the arguments, the model lifecycle, the protocol and the tests. This page covers what the NPU image adds.

## Models

The QNN builds of the out-of-the-box models go in `models/ei-ootb-models/arm64/` here, and only this image
copies them. A brick that needs a QNN model adds it to this folder; the first one also adds the `COPY` line,
since a `COPY` matching no file fails the build.

## Image

From the repository root, bake builds the layer and the parent in-graph:

```bash
docker buildx bake edge-impulse-npu-runner
```
