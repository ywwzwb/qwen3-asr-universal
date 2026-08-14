#!/usr/bin/env bash
# ci/build_linux_container.sh
# Runs inside a python:3.12-slim-bullseye container (Debian 11, glibc >= 2.31)
# on a GitHub-hosted (ubuntu-latest) runner, so the PyInstaller output stays
# compatible with old distros (Ubuntu 20.04+, Debian 11+, ...).
# Usage: TARGET=<target> bash ci/build_linux_container.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Official python images are built with --enable-shared, which PyInstaller
# requires (the manylinux CPython is static and cannot be used).
# objdump (binutils) is needed by PyInstaller to inspect binaries.
apt-get update
apt-get install -y --no-install-recommends binutils >/dev/null

# linux-x64-cuda builds llama-cpp-python from source, which needs the CUDA
# toolkit (nvcc) plus cmake/ninja. The CPU/Vulkan targets use prebuilt wheels.
if [[ "$TARGET" == *-cuda ]]; then
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg cmake ninja-build >/dev/null
    curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb" \
        -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb >/dev/null
    apt-get update
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-4 cuda-cudart-dev-12-4 libcublas-dev-12-4 >/dev/null
    export PATH="/usr/local/cuda-12.4/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
fi

python ci/build.py "$TARGET"
