#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# Unit test for qairt-entrypoint.sh: runs the script against temporary
# directories (via the HOST_QCOM/QCOM overrides) and checks the merged
# /usr/share/qcom view it builds.
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRYPOINT="$SCRIPT_DIR/../qairt-entrypoint.sh"

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

run_entrypoint() { # <host_dir> <qcom_dir> [cmd...]
  # Restrictive umask on purpose: the script must enforce world-readable
  # results (644/755) on its own.
  (umask 077; HOST_QCOM="$1" QCOM="$2" sh "$ENTRYPOINT" "${@:3}")
}

# Baked file the script copies from; point it at a fixture unless running
# inside the image where /etc/fastrpc exists.
BAKED=/etc/fastrpc/hexagon-dsp-binaries.yaml
if [ ! -f "$BAKED" ]; then
  mkdir -p "$WORK/etc-fastrpc"
  BAKED="$WORK/etc-fastrpc/hexagon-dsp-binaries.yaml"
  cat > "$BAKED" <<'EOF'
machines:
  Fake Machine:
    DSP_LIBRARY_PATH: fake/dsp
EOF
  # The script hardcodes /etc/fastrpc: run it through a wrapper that fakes it
  # with a bind of sed. Simpler: substitute the path in a temp copy.
  sed "s|/etc/fastrpc/hexagon-dsp-binaries.yaml|$BAKED|" "$ENTRYPOINT" > "$WORK/entrypoint.sh"
  ENTRYPOINT="$WORK/entrypoint.sh"
fi

echo "== case 1: full host dir (payload + conf.d yamls)"
HOST="$WORK/host1"
QCOM="$WORK/qcom1"
mkdir -p "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp" "$HOST/conf.d"
echo bin > "$HOST/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
cat > "$HOST/conf.d/hexagon-dsp-binaries.yaml" <<'EOF'
machines:
  Host Machine:
    DSP_LIBRARY_PATH: host/dsp
EOF
echo "not-a-config" > "$HOST/conf.d/readme.txt"
run_entrypoint "$HOST" "$QCOM" true

check "payload dir is a symlink to the host mount" test -L "$QCOM/qcs8300"
check "payload files resolve through the symlink" test -f "$QCOM/qcs8300/Qualcomm/QCS8300-RIDE/dsp/fastrpc_shell_3"
check "host yaml copied as a regular file (not symlink)" test -f "$QCOM/conf.d/hexagon-dsp-binaries.yaml"
if [ -L "$QCOM/conf.d/hexagon-dsp-binaries.yaml" ]; then fail "host yaml must not be a symlink"; else pass "host yaml is not a symlink"; fi
check "baked fallback present" test -f "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"
if [ -e "$QCOM/conf.d/readme.txt" ]; then fail "non-yaml host files must not be copied"; else pass "non-yaml host files skipped"; fi
if [ -e "$QCOM/conf.d/conf.d" ]; then fail "conf.d must not be symlinked into itself"; else pass "conf.d not symlinked as payload"; fi
check "conf.d dir is world-traversable (755)" test "$(stat -c %a "$QCOM/conf.d")" = "755"
check "copied host yaml is world-readable (644)" test "$(stat -c %a "$QCOM/conf.d/hexagon-dsp-binaries.yaml")" = "644"
check "baked yaml is world-readable (644)" test "$(stat -c %a "$QCOM/conf.d/00-arduino-dsp-binaries.yaml")" = "644"

echo "== case 2: host dir without conf.d"
HOST="$WORK/host2"
QCOM="$WORK/qcom2"
mkdir -p "$HOST/sa8775p"
run_entrypoint "$HOST" "$QCOM" true
check "no error and baked fallback present" test -f "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"
check "payload still symlinked" test -L "$QCOM/sa8775p"

echo "== case 3: host mount missing entirely"
QCOM="$WORK/qcom3"
run_entrypoint "$WORK/does-not-exist" "$QCOM" true
check "baked fallback is the only conf.d entry" test -f "$QCOM/conf.d/00-arduino-dsp-binaries.yaml"
count=$(find "$QCOM/conf.d" -mindepth 1 | wc -l)
check "conf.d contains exactly one file" test "$count" -eq 1

echo "== case 4: symlinked yaml in host conf.d is not copied"
HOST="$WORK/host4"
QCOM="$WORK/qcom4"
mkdir -p "$HOST/conf.d"
echo "machines: {}" > "$WORK/external.yaml"
ln -s "$WORK/external.yaml" "$HOST/conf.d/linked.yaml"
run_entrypoint "$HOST" "$QCOM" true
if [ -e "$QCOM/conf.d/linked.yaml" ]; then fail "symlinked host yaml must not be copied"; else pass "symlinked host yaml skipped"; fi

echo "== case 5: baked file sorts before typical host yaml names"
first=$(printf '%s\n' "00-arduino-dsp-binaries.yaml" "hexagon-dsp-binaries.yaml" | sort | head -n1)
check "00- prefix sorts first (host wins on last-match)" test "$first" = "00-arduino-dsp-binaries.yaml"

echo "== case 6: exec passthrough"
HOST="$WORK/host6"
QCOM="$WORK/qcom6"
mkdir -p "$HOST"
out=$(run_entrypoint "$HOST" "$QCOM" echo hello-from-cmd)
check "command after setup is exec'd with its args" test "$out" = "hello-from-cmd"

echo "== case 7: idempotent on restart (same container layer)"
run_entrypoint "$WORK/host1" "$WORK/qcom1" true
check "second run succeeds with existing symlinks/files" test -L "$WORK/qcom1/qcs8300"

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES check(s) failed"
  exit 1
fi
echo "All checks passed"
