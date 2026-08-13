#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Builds a merged /usr/share/qcom view in the container layer: DSP payload
# dirs are symlinked from the read-only host mount, conf.d holds copies of
# the host yamls (the fastrpc parser only lists regular files) plus the
# image's baked default. The host mount is read-only: nothing is ever
# written back to the host.
set -eu

# The merged view must stay readable and traversable by every uid, whichever
# user runs the entrypoint (root or arduino): force dirs to 755 and copies
# to 644 regardless of the inherited umask.
umask 022

HOST_QCOM="${HOST_QCOM:-/run/host-qcom}"
QCOM="${QCOM:-/usr/share/qcom}"

mkdir -p "$QCOM/conf.d"

if [ -d "$HOST_QCOM" ]; then
  for entry in "$HOST_QCOM"/*; do
    [ -e "$entry" ] || continue
    name=$(basename "$entry")
    [ "$name" = "conf.d" ] && continue
    ln -sfn "$entry" "$QCOM/$name"
  done
  # conf.d entries must be regular files: fastrpc skips symlinks
  if [ -d "$HOST_QCOM/conf.d" ]; then
    find "$HOST_QCOM/conf.d" -maxdepth 1 -type f \
      \( -name '*.yaml' -o -name '*.yml' \) -exec cp -f {} "$QCOM/conf.d/" \;
  fi
fi

# Sorts first: fastrpc takes the last alphabetical match, so any
# host-provided yaml overrides this fallback.
cp -f /etc/fastrpc/hexagon-dsp-binaries.yaml "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"

exec "$@"
