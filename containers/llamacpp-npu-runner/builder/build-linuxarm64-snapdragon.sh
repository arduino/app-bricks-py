#!/bin/bash
set -euo pipefail

VERSION="${LLAMA_CPP_VERSION:-snapshot}"

cmake --preset arm64-linux-snapdragon-release -B build-snapdragon

cmake --build build-snapdragon -j "$(nproc)"

cmake --install build-snapdragon --prefix pkg-snapdragon

find pkg-snapdragon/bin -maxdepth 1 -name 'test*' -delete

tar -czvf "llamacpp-hexagon-${VERSION}.tgz" pkg-snapdragon
