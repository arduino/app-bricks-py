#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Unit test for provision-fastrpc-dsp.sh: runs the script against temporary
# directories (via the HOST_QCOM/HEXAGON_DSP/BAKED_DSP_YAML overrides) and
# checks the merged /usr/share/hexagon-dsp view it builds.
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/../scripts/provision-fastrpc-dsp.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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

# Baked file the script copies from: use the one shipped by python-base unless
# running inside the image, where /etc/fastrpc exists.
BAKED=/etc/fastrpc/hexagon-dsp-binaries.yaml
if [ ! -f "$BAKED" ]; then
  BAKED="$SCRIPT_DIR/../conf/hexagon-dsp-binaries.yaml"
fi
[ -f "$BAKED" ] || { echo "FAIL: baked yaml not found ($BAKED)"; exit 1; }

run_script() { # <host_dir> <hexagon_dsp_dir>
  # Restrictive umask on purpose: the script must enforce world-readable
  # results (644/755) on its own.
  (umask 077; HOST_QCOM="$1" HEXAGON_DSP="$2" BAKED_DSP_YAML="$BAKED" sh "$SCRIPT")
}

echo "== case 1: full host dir (payload + conf.d yamls)"
HOST="$WORK/host1"
DSP="$WORK/dsp1"
mkdir -p "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp" "$HOST/conf.d"
echo bin > "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
cat > "$HOST/conf.d/hexagon-dsp-binaries.yaml" <<'YAML'
machines:
  Host Machine:
    DSP_LIBRARY_PATH: host/dsp
YAML
echo "not-a-config" > "$HOST/conf.d/readme.txt"
run_script "$HOST" "$DSP"

check "payload dir is a symlink to the host mount" test -L "$DSP/qcs8300"
check "payload files resolve through the symlink" test -f "$DSP/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
check "host yaml copied as a regular file (not symlink)" test -f "$DSP/conf.d/hexagon-dsp-binaries.yaml"
if [ -L "$DSP/conf.d/hexagon-dsp-binaries.yaml" ]; then fail "host yaml must not be a symlink"; else pass "host yaml is not a symlink"; fi
check "baked fallback present" test -f "$DSP/conf.d/00-arduino-dsp-binaries.yaml"
if [ -e "$DSP/conf.d/readme.txt" ]; then fail "non-yaml host files must not be copied"; else pass "non-yaml host files skipped"; fi
if [ -e "$DSP/conf.d/conf.d" ]; then fail "conf.d must not be symlinked into itself"; else pass "conf.d not symlinked as payload"; fi
check "conf.d dir is world-traversable (755)" test "$(stat -c %a "$DSP/conf.d")" = "755"
check "copied host yaml is world-readable (644)" test "$(stat -c %a "$DSP/conf.d/hexagon-dsp-binaries.yaml")" = "644"
check "baked yaml is world-readable (644)" test "$(stat -c %a "$DSP/conf.d/00-arduino-dsp-binaries.yaml")" = "644"

echo "== case 2: host dir without conf.d"
HOST="$WORK/host2"
DSP="$WORK/dsp2"
mkdir -p "$HOST/sa8775p"
run_script "$HOST" "$DSP"
check "no error and baked fallback present" test -f "$DSP/conf.d/00-arduino-dsp-binaries.yaml"
check "payload still symlinked" test -L "$DSP/sa8775p"

echo "== case 3: host mount missing entirely"
DSP="$WORK/dsp3"
run_script "$WORK/does-not-exist" "$DSP"
check "baked fallback is the only conf.d entry" test -f "$DSP/conf.d/00-arduino-dsp-binaries.yaml"
count=$(find "$DSP/conf.d" -mindepth 1 | wc -l)
check "conf.d contains exactly one file" test "$count" -eq 1

echo "== case 4: symlinked yaml in host conf.d is not copied"
HOST="$WORK/host4"
DSP="$WORK/dsp4"
mkdir -p "$HOST/conf.d"
echo "machines: {}" > "$WORK/external.yaml"
ln -s "$WORK/external.yaml" "$HOST/conf.d/linked.yaml"
run_script "$HOST" "$DSP"
if [ -e "$DSP/conf.d/linked.yaml" ]; then fail "symlinked host yaml must not be copied"; else pass "symlinked host yaml skipped"; fi

echo "== case 5: baked file sorts before typical host yaml names"
first=$(printf '%s\n' "00-arduino-dsp-binaries.yaml" "hexagon-dsp-binaries.yaml" | sort | head -n1)
check "00- prefix sorts first (host wins on last-match)" test "$first" = "00-arduino-dsp-binaries.yaml"

echo "== case 6: image without fastrpc (no baked yaml) is a no-op"
DSP="$WORK/dsp6"
(HOST_QCOM="$WORK/host1" HEXAGON_DSP="$DSP" BAKED_DSP_YAML="$WORK/missing.yaml" sh "$SCRIPT")
if [ -e "$DSP" ]; then fail "no view must be built without a baked yaml"; else pass "no baked yaml, nothing provisioned"; fi

echo "== case 7: non-writable view root is skipped, not fatal"
RO="$WORK/readonly"
mkdir -p "$RO"
chmod 555 "$RO"
if [ "$(id -u)" -eq 0 ]; then
  echo "skip: running as root, permissions are not enforced"
else
  out=$(run_script "$WORK/host1" "$RO/hexagon-dsp")
  check "skip is reported" grep -q "not writable" <<<"$out"
fi
chmod 755 "$RO"

echo "== case 8: idempotent on restart (same container layer)"
run_script "$WORK/host1" "$WORK/dsp1"
check "second run succeeds with existing symlinks/files" test -L "$WORK/dsp1/qcs8300"

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES check(s) failed"
  exit 1
fi
echo "All checks passed"
