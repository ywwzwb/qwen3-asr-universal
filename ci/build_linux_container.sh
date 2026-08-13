#!/usr/bin/env bash
# ci/build_linux_container.sh
# Runs inside a pypa/manylinux_2_28 container (glibc >= 2.28) on a GitHub-hosted
# (ubuntu-latest) runner, so the PyInstaller output stays compatible with old
# distros (Ubuntu 18.10+, Debian 10+, ...). Usage: TARGET=<target> bash ci/build_linux_container.sh
set -euo pipefail

# Manylinux images ship CPython 3.12 at /opt/python/cp312-cp312 (pip included).
PY=/opt/python/cp312-cp312/bin/python
"$PY" -m pip install --upgrade pip
"$PY" ci/build.py "$TARGET"
