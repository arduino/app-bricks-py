#!/bin/bash

cmake --preset arm64-linux-snapdragon-release -B build-snapdragon

cmake --build build-snapdragon -j $(nproc)

cmake --install build-snapdragon --prefix pkg-snapdragon
rm -fr pkg-snapdragon/bin/test*

tar -czvf llama-cpp-snapdragon.tar.gz pkg-snapdragon
