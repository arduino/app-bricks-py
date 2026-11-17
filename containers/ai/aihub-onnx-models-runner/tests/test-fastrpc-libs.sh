#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Integration test for the FastRPC client libraries and the QNN execution
# provider shipped by aihub-onnx-models-runner: checks the libraries come from
# the Debian backports package, that both the SONAME and the unversioned names
# load, that the paths the libraries look at are the ones provision-fastrpc-dsp.sh
# and the compose files provide, and that the ONNX Runtime QNN plugin is complete.
#
# Usage: test-fastrpc-libs.sh [image]   (default: app-bricks/aihub-onnx-models-runner:latest)
set -eu

IMAGE="${1:-${ONNX_RUNNER_IMAGE:-app-bricks/aihub-onnx-models-runner:latest}}"
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

# The entrypoint wrapper is what builds the merged DSP view at start-up.
check "the entrypoint wrapper and the provisioning script are installed" \
  in_image 'test -x /aihub-onnx-entrypoint.sh && test -f /provision-fastrpc-dsp.sh'

# The QNN execution provider and the QAIRT it bundles: the wheel's own libQnnHtp.so
# and its DSP-side skels are what a runner uses (utils/onnx_ep.py points
# ADSP_LIBRARY_PATH at them), so both have to be in the image.
check "onnxruntime and the QNN plugin EP are importable" \
  in_image "python -c 'import onnxruntime, onnxruntime_qnn'"
check "the plugin wheel ships its own libQnnHtp.so and its Hexagon skels" \
  in_image "python -c 'import glob, os, onnxruntime_qnn as q; p = q.get_qnn_htp_path(); assert os.path.isfile(p), p; assert glob.glob(os.path.join(os.path.dirname(p), \"libQnnHtpV*Skel.so\"))'"

# The Dockerfile cuts the bundled QAIRT to what the EP loads for HTP v73 and v75: running
# from a context binary needs libQnnSystem.so, compiling a graph needs libQnnHtpPrepare.so,
# and the DSP loads the skel of its own HTP version through ADSP_LIBRARY_PATH.
QNN_DIR=/opt/venv/lib/python3.13/site-packages/onnxruntime_qnn
for lib in libonnxruntime_providers_qnn.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so \
  libQnnHtpV73Stub.so libQnnHtpV73Skel.so libQnnHtpV75Stub.so libQnnHtpV75Skel.so; do
  check "the QNN plugin keeps $lib" in_image "test -f $QNN_DIR/$lib"
done
check "the QNN plugin ships no library the EP never loads" \
  in_image "cd $QNN_DIR && ! ls libGenie.so libQnnGpu.so libQnnIr.so libQnnSaver.so libQnnHtpNetRunExtensions.so libQnnHtpV6[89]* libQnnHtpV79* libQnnHtpV81* 2>/dev/null | grep -q ."

# uv installs the packages, so the venv carries no pip of its own.
check "the venv has no pip of its own" \
  in_image "test ! -e /opt/venv/bin/pip && test -z \"\$(ls -d /opt/venv/lib/python3.13/site-packages/pip* 2>/dev/null)\""

# The QAIRT the wheel bundles is what the shipped HTP context binaries were compiled
# against; pyproject.toml declares it and the python tests compare the two.
declared="$(sed -n 's/^#[[:space:]]*qairt-version:[[:space:]]*//p' "$(dirname "$0")/../pyproject.toml")"
check "the installed wheel bundles the QAIRT declared in pyproject.toml ($declared)" \
  in_image "python -c 'import onnxruntime_qnn as q; assert q.build_and_package_info.qnn_version == \"$declared\", q.build_and_package_info.qnn_version'"

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "All checks passed"
else
  echo "$FAILURES check(s) failed"
  exit 1
fi
