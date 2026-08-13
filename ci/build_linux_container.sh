#!/usr/bin/env bash
# ci/build_linux_container.sh
# Runs inside an `ubuntu:20.04` container on a GitHub-hosted (ubuntu-latest)
# runner. Provides glibc >= 2.31 runtime compatibility for the PyInstaller
# output. Usage: TARGET=<target> bash ci/build_linux_container.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl software-properties-common >/dev/null

add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv >/dev/null

python3.12 -m venv /opt/venv
export PATH=/opt/venv/bin:$PATH
pip install --upgrade pip

python ci/build.py "$TARGET"
