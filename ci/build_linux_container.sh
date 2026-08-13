#!/usr/bin/env bash
# ci/build_linux_container.sh
# Runs inside a python:3.12-slim-bullseye container (Debian 11, glibc >= 2.31)
# on a GitHub-hosted (ubuntu-latest) runner, so the PyInstaller output stays
# compatible with old distros (Ubuntu 20.04+, Debian 11+, ...).
# Usage: TARGET=<target> bash ci/build_linux_container.sh
set -euo pipefail

# Official python images are built with --enable-shared, which PyInstaller
# requires (the manylinux CPython is static and cannot be used).
python ci/build.py "$TARGET"
