#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Usage: make-venv <name> <pip install arguments...>
# Creates /venvs/<name> keeping only the package metadata licensed reads

set -e

name="$1"
shift
venv="/venvs/${name}"

python -m venv "${venv}"
"${venv}/bin/pip" install -q "$@"
find "${venv}"/lib/python*/site-packages -mindepth 1 -maxdepth 1 \
    ! -name '*.dist-info' \
    ! -name 'pip' ! -name 'setuptools' ! -name 'wheel' \
    ! -name 'pkg_resources' ! -name '_distutils_hack' \
    -exec rm -rf {} +
