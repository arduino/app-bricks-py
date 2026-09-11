#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Builds a merged /usr/share/hexagon-dsp view in the container layer: DSP
# payload dirs are symlinked from the host mount, conf.d holds copies of the
# host yamls (the fastrpc parser only lists regular files) plus the image's
# baked default. Nothing is ever written back to the host mount.
#
# Same idea as qairt-entrypoint.sh, but for the Debian libfastrpc1 shipped by
# python-base: that build reads its config from /usr/share/hexagon-dsp/conf.d
# and resolves each DSP_LIBRARY_PATH relative to /usr/share/hexagon-dsp (the
# QAIRT source build uses /usr/share/qcom for both).
set -eu

# The merged view must stay readable and traversable by every uid, whichever
# user runs the app: force dirs to 755 and copies to 644 regardless of the
# inherited umask.
umask 022

# The host DSP installation is bind-mounted read-only at /run/host-qcom, conf.d
# included; the board model stays at its own path and is read by libfastrpc
# directly, so nothing has to be provisioned for it.
HOST_QCOM="${HOST_QCOM:-/run/host-qcom}"
HEXAGON_DSP="${HEXAGON_DSP:-/usr/share/hexagon-dsp}"
BAKED_DSP_YAML="${BAKED_DSP_YAML:-/etc/fastrpc/hexagon-dsp-binaries.yaml}"

# Images built without fastrpc have nothing to provision.
[ -f "$BAKED_DSP_YAML" ] || exit 0

if ! mkdir -p "$HEXAGON_DSP/conf.d" 2>/dev/null; then
  echo "Skipping fastrpc DSP provisioning: $HEXAGON_DSP is not writable"
  exit 0
fi

if [ -d "$HOST_QCOM" ]; then
  echo "Provisioning fastrpc DSP configurations..."
  for entry in "$HOST_QCOM"/*; do
    [ -e "$entry" ] || continue
    name=$(basename "$entry")
    [ "$name" = "conf.d" ] && continue
    ln -sfn "$entry" "$HEXAGON_DSP/$name"
  done
  # conf.d entries must be regular files: fastrpc skips symlinks
  if [ -d "$HOST_QCOM/conf.d" ]; then
    find "$HOST_QCOM/conf.d" -maxdepth 1 -type f \
      \( -name '*.yaml' -o -name '*.yml' \) -exec cp -f {} "$HEXAGON_DSP/conf.d/" \;
  fi
fi

# Sorts first: fastrpc takes the last alphabetical match, so any host-provided
# yaml overrides this fallback.
cp -f "$BAKED_DSP_YAML" "$HEXAGON_DSP/conf.d/00-arduino-dsp-binaries.yaml"
