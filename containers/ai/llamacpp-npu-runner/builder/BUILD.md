# Compile instructions

It must be cross compiled from an amd64 host.

Here are instructions to compile llama.cpp with Hexagon backend.

1) Clone llama.cpp repo locally on an amd64 host
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout ${LLAMA_CPP_VERSION}
cp ../CMakeUserPresets.json .
cp ../build-linuxarm64-snapdragon.sh .
```

2) Start sdk container
```bash
docker run -it -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-linux:v0.1
cd /workspace
```

3)  Start build
```bash
./build-linuxarm64-snapdragon.sh
```
