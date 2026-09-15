#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Integration test for the FastRPC client libraries shipped by python-base:
# checks they come from the Debian backports package, that both the SONAME and
# the unversioned names load, and that the paths the libraries look at are the
# ones the compose files are expected to provide.
#
# Usage: test-fastrpc-libs.sh [image]   (default: app-bricks/python-base:latest)
set -eu

IMAGE="${1:-${PYTHON_BASE_IMAGE:-app-bricks/python-base:latest}}"
PLATFORM="${PLATFORM:-linux/arm64}"

FAILURES=0

fail() {
  echo "FAIL: $1"
  FAILURES=$((FAILURES + 1))
}

pass() {
  echo "ok:   $1"
}

check() { # <description> <condition...>
  local desc="$1"
  shift
  if "$@"; then pass "$desc"; else fail "$desc"; fi
}

in_image() { # <shell snippet>
  docker run --rm --platform "$PLATFORM" --entrypoint sh "$IMAGE" -c "$1"
}

echo "Testing $IMAGE ($PLATFORM)"

# The package must be the Qualcomm-maintained Debian one from trixie-backports,
# not a local source build: a ~bpo13 version is the marker.
version="$(in_image 'dpkg -s libfastrpc1 2>/dev/null | sed -n "s/^Version: //p"')"
echo "libfastrpc1 version: ${version:-<not installed>}"
check "libfastrpc1 installed from trixie-backports" \
  grep -q 'bpo13' <<<"$version"

# libtranslation.so and the QNN HTP backend link against the SONAMEs, while
# code that dlopens the libraries by plain name needs the unversioned symlinks
# (which live in libfastrpc-dev, so python-base recreates them).
for lib in libcdsprpc libadsprpc libsdsprpc; do
  check "$lib.so.1 is in the ldconfig cache" \
    in_image "ldconfig -p | grep -q '$lib\.so\.1 '"
  check "$lib.so and $lib.so.1 both load" \
    in_image "python -c \"import ctypes; ctypes.CDLL('$lib.so'); ctypes.CDLL('$lib.so.1')\""
done

# The Debian build keeps the upstream paths: the machine name comes from the
# device tree directly (no /run/device-model bind mount) and the DSP config
# from /usr/share/hexagon-dsp/conf.d (not /usr/share/qcom). Guard them: a
# switch back to a patched source build would silently move both.
check "machine name is read from the device tree" \
  in_image 'grep -qa "/sys/firmware/devicetree/base/model" /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1'
check "DSP config is read from /usr/share/hexagon-dsp/conf.d" \
  in_image 'grep -qa "/usr/share/hexagon-dsp/conf.d/" /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1'

# The backports source is only meant to exist during the build.
check "no backports apt source left in the image" \
  in_image '! ls /etc/apt/sources.list.d/ | grep -q backports'

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "All checks passed"
else
  echo "$FAILURES check(s) failed"
  exit 1
fi
