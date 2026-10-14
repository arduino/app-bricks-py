#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Entrypoint wrapper for the ONNX/QNN runners: provisions the Hexagon DSP view
# the FastRPC client libraries need, then execs the app.
#
# The counterpart of qairt-common-base's /qairt-entrypoint.sh. It is a separate
# script because this image ships the Debian libfastrpc1, which reads its config
# from /usr/share/hexagon-dsp/conf.d rather than /usr/share/qcom.
set -eu

# Best-effort: a failure here only means the NPU is unavailable and the QNN
# execution provider falls back to the CPU, so the app still starts.
sh /provision-fastrpc-dsp.sh || echo "Warning: fastrpc DSP provisioning failed"

exec "$@"
