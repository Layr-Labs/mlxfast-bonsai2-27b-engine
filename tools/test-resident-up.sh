#!/usr/bin/env bash
# test-resident-up.sh -- prove tools/resident-up.sh without a model, a GPU or
# a network.
#
# WHAT THIS PROVES, with a stub bench-worker that binds its socket after a
# delay and answers the hello the way the real resident does:
#   1. a served window: the stub receives the contract argv (resident,
#      --weights, --speculative-protocol v1.1, --socket), the wrapped
#      command sees BENCH_WORKER_RESIDENT_SOCKET and the resident's identity
#      file, its exit code is returned, and after the window the stub, the
#      socket, the pidfile and the pgid file are all gone
#   2. --hello-identity reaches the stub and the command
#   3. the GPU lock must be HELD: nobody holding it refuses by name before any
#      boot; the refusal has no environment bypass
#   4. an unpinned iogpu wired limit refuses by name before any boot
#   5. a resident already up (live pidfile, or a socket that answers) refuses
#      by name, and the first window is left alone
#   6. a resident that dies before it is healthy fails the window, with its log
#   7. a resident that never binds fails on the health timeout
#   8. a resident that ignores SIGTERM is killed with the process group
#
# And the PER-LEG form benchd drives (--boot / --stop), against a synthetic
# tree:
#   9.  --boot writes the socket path as the first line of --socket-out, a
#       <socket>.pid sidecar and a <socket>.ready marker, exits 0, and leaves
#       the resident RUNNING
#   10. --spec serial is authoritative over a tree whose mtp-head.manifest.json
#       declares a draft depth
#   11. the boot loads THAT TREE'S own weights, not an inherited path
#   12. a second --boot while a resident is up refuses box-wide, across trees
#   13. --stop ends the resident and removes all three files; a second --stop is
#       a no-op; the next leg boots afterwards
#   14. every flag without a value, every missing required flag, and every
#       spec/draft-length mismatch refuses BY NAME
#
# Hermetic: bash (3.2 on macOS is enough) and python3.
#
# Usage: tools/test-resident-up.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
RESIDENT_UP="${ROOT_DIR}/tools/resident-up.sh"

WORK="$(mktemp -d)"
trap 'cleanup' EXIT
LOCK_HOLDER_PID=""
cleanup() {
  [[ -n "${LOCK_HOLDER_PID}" ]] && kill "${LOCK_HOLDER_PID}" 2>/dev/null
  pkill -f "stub-bench-worker.py" 2>/dev/null
  # The per-leg cases stage the stub under the name benchd resolves, so it does
  # not match the pattern above.
  pkill -f "release/bench-worker resident" 2>/dev/null
  rm -rf "${WORK}"
}

fails=0
pass() { printf 'test-resident-up: PASS -- %s\n' "$*"; }
fail() { printf 'test-resident-up: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

# --- the stub resident --------------------------------------------------------
# Speaks the contract: validates argv, waits STUB_LOAD_S, binds the socket,
# then serves one connection at a time: on accept it sends the hello line
# (unprompted, ok:true, backend mlx-resident) and closes when the client does.
# STUB_DIE_BEFORE_BIND=1 exits 3 before binding. STUB_NEVER_BIND=1 sleeps
# forever without binding. STUB_IGNORE_TERM=1 ignores SIGTERM.
# It records its argv in STUB_ARGV_FILE.
STUB="${WORK}/stub-bench-worker.py"
cat > "${STUB}" <<'PY'
#!/usr/bin/env python3
import json, os, signal, socket, sys, time
argv = sys.argv[1:]
with open(os.environ["STUB_ARGV_FILE"], "w") as f:
    f.write("\n".join(argv) + "\n")
if not argv or argv[0] != "resident":
    print("stub: first argument must be resident", file=sys.stderr); sys.exit(2)
opts = {}
i = 1
while i < len(argv):
    if argv[i].startswith("--"):
        opts.setdefault(argv[i], []).append(argv[i + 1] if i + 1 < len(argv) else None); i += 2
    else:
        print(f"stub: stray argument {argv[i]}", file=sys.stderr); sys.exit(2)
for needed in ("--weights", "--drafter", "--socket", "--speculative-protocol"):
    if needed not in opts:
        print(f"stub: missing {needed}", file=sys.stderr); sys.exit(2)
if opts["--speculative-protocol"] != ["v1.1"]:
    print("stub: --speculative-protocol must be v1.1", file=sys.stderr); sys.exit(2)
if os.environ.get("STUB_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
time.sleep(float(os.environ.get("STUB_LOAD_S", "1")))
if os.environ.get("STUB_DIE_BEFORE_BIND") == "1":
    print("stub: dying before bind, as asked", file=sys.stderr); sys.exit(3)
if os.environ.get("STUB_NEVER_BIND") == "1":
    while True:
        time.sleep(1)
path = opts["--socket"][0]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
hello = {"id": 0, "nonce": "stubnonce", "ok": True, "protocol_version": 1,
         "backend": "mlx-resident", "device": "stub", "spec_modes": ["serial", "mtp", "dflash"],
         "runner": {"id": "layr/stub"}, "resident": {"pid": os.getpid(), "load_epoch": 1}}
while True:
    conn, _ = srv.accept()
    try:
        conn.sendall((json.dumps(hello) + "\n").encode())
        while conn.recv(65536):
            pass
    except OSError:
        pass
    finally:
        conn.close()
PY
chmod +x "${STUB}"

# --- the lock, held by a background holder for the served cases --------------
LOCK="${WORK}/gpu.lock"
: > "${LOCK}"
hold_lock() {
  python3 - "${LOCK}" <<'PY' &
import fcntl, sys, time
f = open(sys.argv[1], "a+")
fcntl.flock(f, fcntl.LOCK_EX)
while True:
    time.sleep(1)
PY
  LOCK_HOLDER_PID=$!
  sleep 0.5
}
release_lock() {
  [[ -n "${LOCK_HOLDER_PID}" ]] && kill "${LOCK_HOLDER_PID}" 2>/dev/null
  wait "${LOCK_HOLDER_PID}" 2>/dev/null
  LOCK_HOLDER_PID=""
}

WEIGHTS="${WORK}/weights"; mkdir -p "${WEIGHTS}"
# THE HEAD IS A SEPARATE ARTIFACT on this track, so the boot resolves it and
# refuses without it. Every case below stages one; the case that proves the
# refusal points MLXFAST_MTP_HEAD_DIR at a path that does not exist.
HEAD="${WORK}/mtp-head"; mkdir -p "${HEAD}"
export MLXFAST_MTP_HEAD_DIR="${HEAD}"
WIRED_OK="${WORK}/wired-ok.sh"; printf '#!/bin/sh\necho 112000\n' > "${WIRED_OK}"; chmod +x "${WIRED_OK}"
WIRED_ZERO="${WORK}/wired-zero.sh"; printf '#!/bin/sh\necho 0\n' > "${WIRED_ZERO}"; chmod +x "${WIRED_ZERO}"

# run_up LOGDIR [ENV=VAL...] -- ARGS... : drive the real script with the stub.
run_up() {
  local logdir="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  # The case's own ENV=VAL pairs come LAST, so a case can override a default.
  env MLXFAST_ENGINE_BIN="${STUB}" \
    RESIDENT_UP_LOG_DIR="${logdir}" \
    RESIDENT_UP_LOCK_PATH="${LOCK}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    STUB_ARGV_FILE="${logdir}/stub-argv" \
    ${envs[@]+"${envs[@]}"} \
    "${RESIDENT_UP}" "$@" 2>&1
}

# --- 1. a served window ------------------------------------------------------
hold_lock
L1="${WORK}/case1"; mkdir -p "${L1}"
SOCK1="${WORK}/case1.sock"
out="$(run_up "${L1}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK1}" -- \
  bash -c 'echo "socket=$BENCH_WORKER_RESIDENT_SOCKET hello=${BENCH_WORKER_RESIDENT_HELLO:-unset} identity=$RESIDENT_IDENTITY_FILE"; test -S "$BENCH_WORKER_RESIDENT_SOCKET" && echo socket-live; kill -0 "$RESIDENT_UP_PID" && echo resident-live; exit 7')"
rc=$?
if [[ "${rc}" -eq 7 ]]; then pass "the wrapped command's exit code (7) is returned"; else fail "expected exit 7, got ${rc}: ${out}"; fi
if [[ "${out}" == *"socket=${SOCK1} hello=unset identity=${L1}/resident-identity.json"* && "${out}" == *"socket-live"* && "${out}" == *"resident-live"* ]]; then
  pass "the command runs with BENCH_WORKER_RESIDENT_SOCKET, no hello identity by default, and a live resident"
else
  fail "the command did not see the window: ${out}"
fi
if [[ -f "${L1}/stub-argv" ]] && grep -qx -- "resident" "${L1}/stub-argv" && grep -qx -- "--speculative-protocol" "${L1}/stub-argv" \
   && grep -qx -- "v1.1" "${L1}/stub-argv" \
   && grep -qx -- "${SOCK1}" "${L1}/stub-argv"; then
  pass "the resident received the contract argv"
else
  fail "the resident argv is wrong: $(tr '\n' ' ' < "${L1}/stub-argv" 2>/dev/null)"
fi
if [[ "${out}" == *"resident healthy on ${SOCK1}"* ]]; then pass "the health probe read the resident's hello"; else fail "no health line: ${out}"; fi
if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["hello"]["backend"]=="mlx-resident" and d["weight_owner"]=="bench-worker-resident"' "${L1}/resident-identity.json" 2>/dev/null; then
  pass "the identity file records the resident's hello"
else
  fail "identity file missing or wrong: $(cat "${L1}/resident-identity.json" 2>/dev/null)"
fi
sleep 0.5
if [[ ! -e "${SOCK1}" && ! -e "${L1}/resident.pid" && ! -e "${L1}/resident.pgid" ]] && ! pgrep -f "stub-bench-worker.py resident --weights ${WEIGHTS} --speculative-protocol v1.1 --socket ${SOCK1}" >/dev/null; then
  pass "after the window: resident gone, socket, pidfile and pgid file removed"
else
  fail "the window left something behind: $(ls "${L1}" "${SOCK1}" 2>&1 | tr '\n' ' ')"
fi

# --- 2. --hello-identity -----------------------------------------------------
L2="${WORK}/case2"; mkdir -p "${L2}"
SOCK2="${WORK}/case2.sock"
out="$(run_up "${L2}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --hello-identity --socket "${SOCK2}" -- \
  bash -c 'echo "hello=${BENCH_WORKER_RESIDENT_HELLO:-unset}"')"
if [[ "${out}" == *"hello=1"* ]]; then
  pass "--hello-identity exports BENCH_WORKER_RESIDENT_HELLO=1"
else
  fail "--hello-identity not honoured: ${out} / $(tr '\n' ' ' < "${L2}/stub-argv" 2>/dev/null)"
fi

# --- 5. a resident already up refuses ----------------------------------------
L5="${WORK}/case5"; mkdir -p "${L5}"
SOCK5="${WORK}/case5.sock"
FIRST_OUT="${WORK}/case5.first"
( run_up "${L5}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- \
    bash -c 'sleep 6; echo first-window-finished' > "${FIRST_OUT}" 2>&1 ) &
FIRST_PID=$!
for _ in $(seq 1 40); do [[ -S "${SOCK5}" ]] && break; sleep 0.2; done
out="$(run_up "${L5}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- true)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* && "${out}" == *"Nothing has been loaded"* ]]; then
  pass "a second window while a resident is up refuses by name (same pidfile)"
else
  fail "a second window did not refuse by name (rc ${rc}): ${out}"
fi
L5b="${WORK}/case5b"; mkdir -p "${L5b}"
out="$(run_up "${L5b}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- true)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* && "${out}" == *"already answers on ${SOCK5}"* ]]; then
  pass "a second window against a socket that answers refuses by name (different log dir)"
else
  fail "a socket that answers did not refuse (rc ${rc}): ${out}"
fi
wait "${FIRST_PID}"
if grep -q "first-window-finished" "${FIRST_OUT}"; then
  pass "the first window was left alone and finished"
else
  fail "the first window was disturbed: $(cat "${FIRST_OUT}")"
fi

# --- 6. a resident that dies before it is healthy ----------------------------
L6="${WORK}/case6"; mkdir -p "${L6}"
out="$(run_up "${L6}" STUB_DIE_BEFORE_BIND=1 RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${WORK}/case6.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"exited before it was healthy"* && "${out}" == *"dying before bind"* && "${out}" != *"should-not-run"* ]]; then
  pass "a resident that dies before it is healthy fails the window with its log, and the command never runs"
else
  fail "an early death was not reported (rc ${rc}): ${out}"
fi

# --- 7. a resident that never binds ------------------------------------------
L7="${WORK}/case7"; mkdir -p "${L7}"
out="$(run_up "${L7}" STUB_NEVER_BIND=1 RESIDENT_UP_HEALTH_TIMEOUT_S=3 --weights "${WEIGHTS}" --socket "${WORK}/case7.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"not healthy within 3s"* && "${out}" != *"should-not-run"* ]] && ! pgrep -f "case7.sock" >/dev/null; then
  pass "a resident that never binds fails on the health timeout and is torn down"
else
  fail "the health timeout did not fire or the resident survived (rc ${rc}): ${out}"
fi

# --- 8. a resident that ignores SIGTERM --------------------------------------
L8="${WORK}/case8"; mkdir -p "${L8}"
out="$(run_up "${L8}" STUB_IGNORE_TERM=1 RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${WORK}/case8.sock" -- true)"; rc=$?
sleep 0.5
if [[ "${rc}" -eq 0 && "${out}" == *"killing process group"* ]] && ! pgrep -f "case8.sock" >/dev/null; then
  pass "a resident that ignores SIGTERM is killed with its process group; nothing survives"
else
  fail "SIGTERM escalation failed (rc ${rc}): ${out}; survivors: $(pgrep -fl case8.sock | tr '\n' ' ')"
fi

release_lock

# --- 3. the lock must be held -------------------------------------------------
L3="${WORK}/case3"; mkdir -p "${L3}"
out="$(run_up "${L3}" --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-not-held)"* && "${out}" == *"Nothing has been loaded"* && ! -f "${L3}/stub-argv" ]]; then
  pass "nobody holding the GPU lock refuses by name before any boot"
else
  fail "an unheld lock did not refuse (rc ${rc}): ${out}"
fi
out="$(run_up "${L3}" RESIDENT_UP_SKIP_LOCK=1 RESIDENT_UP_LOCK_CHECK=0 --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-not-held)"* ]]; then
  pass "the lock refusal has no environment bypass"
else
  fail "an environment variable relaxed the lock refusal (rc ${rc}): ${out}"
fi
out="$(run_up "${L3}" RESIDENT_UP_LOCK_PATH="${WORK}/no-such-lock" --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-missing)"* ]]; then
  pass "a missing lock file refuses by name"
else
  fail "a missing lock file did not refuse (rc ${rc}): ${out}"
fi

# --- 4. the wired limit must be pinned ---------------------------------------
hold_lock
L4="${WORK}/case4"; mkdir -p "${L4}"
out="$(run_up "${L4}" RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_ZERO}" --weights "${WEIGHTS}" --socket "${WORK}/case4.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (wired-limit-unpinned)"* && ! -f "${L4}/stub-argv" ]]; then
  pass "an unpinned iogpu wired limit refuses by name before any boot"
else
  fail "an unpinned wired limit did not refuse (rc ${rc}): ${out}"
fi
release_lock

# --- argv refusals -------------------------------------------------------------
# THE HEAD REFUSAL. A boot without a staged head names the head and says how
# to stage it, rather than loading a target with no drafter and reporting a
# serial residency for a speculative leg.
out="$(MLXFAST_MTP_HEAD_DIR="${WORK}/no-such-head" "${RESIDENT_UP}" --weights "${WEIGHTS}" --socket "${WORK}/case-head.sock" -- true 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"MTP head directory is missing"* ]]; then
  pass "a missing MTP head refuses by name"
else
  fail "missing head (rc ${rc}): ${out}"
fi

out="$("${RESIDENT_UP}" --weights "${WEIGHTS}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"no command after '--'"* ]]; then pass "a missing command refuses"; else fail "missing command (rc ${rc}): ${out}"; fi
out="$("${RESIDENT_UP}" -- true 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--weights <dir> is required"* ]]; then pass "a missing --weights refuses"; else fail "missing weights (rc ${rc}): ${out}"; fi


# ===========================================================================
# THE PER-LEG FORM: --boot / --stop.
#
# benchd calls these once per leg, in THAT LEG'S OWN TREE, so the cases below
# build a synthetic tree: a copy of the real script, the stub staged at the
# fixed path benchd resolves (.build/release/bench-worker), the tree's own
# weights/, and an mtp-head.manifest.json declaring a draft depth -- which
# --spec serial must ignore.
# ===========================================================================
hold_lock

TREE="${WORK}/leg-tree"
mkdir -p "${TREE}/tools" "${TREE}/.build/release" "${TREE}/weights" \
  "${TREE}/reference_weights/Qwen3.8-27B-MTP-4bit" \
  "${TREE}/reference_weights/Qwen3.8-27B-DFlash2"
cp "${RESIDENT_UP}" "${TREE}/tools/resident-up.sh"
chmod +x "${TREE}/tools/resident-up.sh"
# The boot reads the leg's DECLARED decoder through the tree's own trusted
# reader, so the synthetic tree carries that reader and the contract it checks
# the declaration against.
mkdir -p "${TREE}/fixtures"
cp "${ROOT_DIR}/tools/spec-declaration.sh" "${TREE}/tools/spec-declaration.sh"
chmod +x "${TREE}/tools/spec-declaration.sh"
cp "${ROOT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json" "${TREE}/fixtures/"
# The stub is staged UNDER THE NAME benchd resolves, so the box-wide
# "bench-worker resident" scan sees it exactly as it sees a real one.
cp "${STUB}" "${TREE}/.build/release/bench-worker"
chmod +x "${TREE}/.build/release/bench-worker"
# A manifest that declares a SPECULATIVE tree. --spec serial must not read it.
cat > "${TREE}/mtp-head.manifest.json" <<'MANIFEST'
{
  "source": "pinned",
  "spec": {
    "enabled": true,
    "num_speculative_tokens": 3
  }
}
MANIFEST

TREE_UP="${TREE}/tools/resident-up.sh"
BOOT_LOG="${WORK}/boot-log"; mkdir -p "${BOOT_LOG}"

# run_leg [ENV=VAL...] -- ARGS... : drive the tree's own copy.
run_leg() {
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  env -u MLXFAST_ENGINE_BIN \
    RESIDENT_UP_LOG_DIR="${BOOT_LOG}" \
    RESIDENT_UP_LOCK_PATH="${LOCK}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    RESIDENT_UP_HEALTH_TIMEOUT_S=30 \
    STUB_ARGV_FILE="${BOOT_LOG}/stub-argv" \
    ${envs[@]+"${envs[@]}"} \
    "${TREE_UP}" "$@" 2>&1
}

# --- B1. --boot reports the socket, writes the sidecars, and leaves it up ----
SOCKET_OUT="${WORK}/leg1.socket"
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${SOCKET_OUT}")"; rc=$?
boot_socket="$(head -n 1 "${SOCKET_OUT}" 2>/dev/null || true)"
if [[ "${rc}" -eq 0 && -n "${boot_socket}" && -S "${boot_socket}" ]]; then
  pass "--boot exits 0 and writes the socket path as the FIRST line of --socket-out"
else
  fail "--boot did not report a live socket (rc ${rc}, socket '${boot_socket}'): ${out}"
fi
boot_pid="$(cat "${boot_socket}.pid" 2>/dev/null || true)"
if [[ -f "${boot_socket}.pid" && "${boot_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${boot_pid}" 2>/dev/null; then
  pass "--boot writes a <socket>.pid sidecar naming a LIVE resident (it is still running after the boot exits)"
else
  fail "--boot left no live pid sidecar (pid '${boot_pid}'): ${out}"
fi
if [[ -f "${boot_socket}.ready" ]]; then
  pass "--boot writes the <socket>.ready marker"
else
  fail "--boot wrote no ready marker beside ${boot_socket}"
fi

# --- B2. --spec serial is authoritative over the tree's manifest -------------
# The tree declares depth 3. benchd asked for the SERIAL control leg. The
# manifest must not reach the boot at all.
if grep -q '^serial 0 ' "${boot_socket}.ready" \
   && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["spec"]=="serial" and d["draft_len"]=="0" else 1)' \
        "${BOOT_LOG}/resident-identity.json" 2>/dev/null; then
  pass "--spec serial is authoritative: the tree's mtp-head.manifest.json declares depth 3 and the boot records serial 0"
else
  fail "--spec serial did not win over the manifest: ready='$(cat "${boot_socket}.ready" 2>/dev/null)' identity=$(cat "${BOOT_LOG}/resident-identity.json" 2>/dev/null | tr '\n' ' ')"
fi

# --- B3. the boot ran THIS TREE'S worker, not an inherited MLXFAST_ENGINE_BIN
if grep -q "${TREE}/reference_weights/Qwen3.8-27B-MTP-4bit" "${BOOT_LOG}/stub-argv" 2>/dev/null; then
  pass "--boot named the tree's OWN MTP head (both legs load the same bytes)"
else
  fail "--boot did not pass --drafter: $(tr '\n' ' ' < "${BOOT_LOG}/stub-argv" 2>/dev/null)"
fi
if grep -q "${TREE}/weights" "${BOOT_LOG}/stub-argv" 2>/dev/null; then
  pass "--boot loaded the tree's OWN weights/ (a leg is defined by its tree)"
else
  fail "--boot did not use the tree's own weights: $(tr '\n' ' ' < "${BOOT_LOG}/stub-argv" 2>/dev/null)"
fi

# --- B4. a second boot while one is up refuses, box-wide ---------------------
# The two legs live in two trees with two pidfiles, so only a box-wide scan
# catches leg 2 booting before leg 1 was stopped -- which would double-load.
out="$(run_leg --boot --spec mtp --draft-len 2 --socket-out "${WORK}/leg2.socket")"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* ]]; then
  pass "a second --boot while a resident is up refuses by name (box-wide, across trees)"
else
  fail "a second --boot did not refuse (rc ${rc}): ${out}"
fi

# --- B5. --stop ends the resident and removes all three files ----------------
out="$(run_leg --stop --socket "${boot_socket}")"; rc=$?
sleep 0.5
if [[ "${rc}" -eq 0 ]] && ! kill -0 "${boot_pid}" 2>/dev/null \
   && [[ ! -e "${boot_socket}" && ! -e "${boot_socket}.pid" && ! -e "${boot_socket}.ready" ]]; then
  pass "--stop ends the resident and removes the socket, the pid sidecar and the ready marker"
else
  fail "--stop left something behind (rc ${rc}): $(ls "${boot_socket}"* 2>&1 | tr '\n' ' ') ${out}"
fi

# --- B6. --stop is idempotent ------------------------------------------------
out="$(run_leg --stop --socket "${boot_socket}")"; rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"nothing to stop"* ]]; then
  pass "a second --stop is a no-op: exit 0, and it says there was nothing to stop"
else
  fail "a second --stop was not a no-op (rc ${rc}): ${out}"
fi

# --- B7. a boot after a stop works: the box-wide scan sees a clean box -------
SOCKET_OUT2="${WORK}/leg2.socket"
out="$(run_leg --boot --spec mtp --draft-len 4 --socket-out "${SOCKET_OUT2}")"; rc=$?
boot2_socket="$(head -n 1 "${SOCKET_OUT2}" 2>/dev/null || true)"
if [[ "${rc}" -eq 0 && -S "${boot2_socket}" ]] && grep -q '^mtp 4 ' "${boot2_socket}.ready"; then
  pass "the next leg boots after a stop, and records its own spec (mtp 4)"
else
  fail "the second leg did not boot (rc ${rc}): ${out}"
fi
run_leg --stop --socket "${boot2_socket}" >/dev/null 2>&1

# --- B7b. the DRAFTER comes from the tree's declaration, not from --spec ----
# benchd derives the boot label from `spec.mtp.depth` alone, so a DFlash
# candidate is labelled `serial 0` too. A drafter chosen from the label would
# then be the MTP head, the worker would advertise `mtp` only, and benchd would
# refuse the dflash request. The tree's own declaration decides instead.
cat > "${TREE}/mtp-head.manifest.json" <<'DFLASHMANIFEST'
{
  "source": "pinned",
  "spec": {
    "decoder": "dflash",
    "enabled": true,
    "num_speculative_tokens": 7
  }
}
DFLASHMANIFEST
rm -f "${BOOT_LOG}/stub-argv"
DFLASH_SOCKET_OUT="${WORK}/leg-dflash.socket"
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${DFLASH_SOCKET_OUT}")"; rc=$?
dflash_socket="$(head -n 1 "${DFLASH_SOCKET_OUT}" 2>/dev/null || true)"
if [[ "${rc}" -eq 0 ]] \
   && grep -q "${TREE}/reference_weights/Qwen3.8-27B-DFlash2" "${BOOT_LOG}/stub-argv" 2>/dev/null; then
  pass "a tree that DECLARES dflash boots the DFlash 2 drafter even when benchd labels the leg serial"
else
  fail "a dflash declaration did not select the DFlash 2 drafter (rc ${rc}): $(tr '\n' ' ' < "${BOOT_LOG}/stub-argv" 2>/dev/null)"
fi
# The LABEL is still the label: the leg records what benchd asked for.
if grep -q '^serial 0 ' "${dflash_socket}.ready" 2>/dev/null; then
  pass "--spec still decides what the leg IS: the dflash-declaring tree records serial 0"
else
  fail "the dflash-declaring tree did not record the label benchd passed: $(cat "${dflash_socket}.ready" 2>/dev/null)"
fi
run_leg --stop --socket "${dflash_socket}" >/dev/null 2>&1 || true

# A declaration the trusted reader refuses is a refusal here, never a guess.
cat > "${TREE}/mtp-head.manifest.json" <<'BADMANIFEST'
{ "source": "pinned", "spec": { "decoder": "dspark", "enabled": true, "num_speculative_tokens": 1 } }
BADMANIFEST
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${WORK}/decoder-bad" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"declaration-invalid"* ]]; then
  pass "a declaration the trusted reader refuses stops the boot by name"
else
  fail "a refused declaration did not stop the boot (rc ${rc}): ${out}"
fi

# Put the speculative MTP declaration back for the cases that follow.
cat > "${TREE}/mtp-head.manifest.json" <<'MANIFEST'
{
  "source": "pinned",
  "spec": {
    "enabled": true,
    "num_speculative_tokens": 3
  }
}
MANIFEST

out="$("${TREE_UP}" --boot --spec dflash --draft-len 17 --socket-out "${WORK}/dflash-deep" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--spec dflash requires --draft-len between 1 and 16"* ]]; then
  pass "--spec dflash refuses a draft length above 16"
else
  fail "--spec dflash did not refuse depth 17 (rc ${rc}): ${out}"
fi

out="$("${TREE_UP}" --boot --spec mtp --draft-len 16 --socket-out "${WORK}/mtp-deep" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--spec mtp requires --draft-len between 1 and 7"* ]]; then
  pass "--spec mtp keeps its own 1..7 envelope; the two decoders do not share one"
else
  fail "--spec mtp did not refuse depth 16 (rc ${rc}): ${out}"
fi

# --- B8. a flag without a value refuses BY NAME ------------------------------
for flag in --spec --draft-len --socket-out --socket; do
  out="$("${TREE_UP}" --boot "${flag}" 2>&1)"; rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (bad-argument)"* && "${out}" == *"${flag} needs"* ]]; then
    pass "${flag} without a value refuses by name"
  else
    fail "${flag} without a value did not refuse by name (rc ${rc}): ${out}"
  fi
done

# --- B9. --boot needs all three, and each refusal names the missing one ------
out="$("${TREE_UP}" --boot --draft-len 0 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --spec"* ]]; then pass "--boot without --spec refuses by name"; else fail "--boot without --spec (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --boot --spec serial --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --draft-len"* ]]; then pass "--boot without --draft-len refuses by name"; else fail "--boot without --draft-len (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --boot --spec serial --draft-len 0 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --socket-out"* ]]; then pass "--boot without --socket-out refuses by name"; else fail "--boot without --socket-out (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --stop 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--stop needs --socket"* ]]; then pass "--stop without --socket refuses by name"; else fail "--stop without --socket (rc ${rc}): ${out}"; fi

# --- B10. spec and draft length must agree ----------------------------------
out="$("${TREE_UP}" --boot --spec serial --draft-len 3 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"serial requires --draft-len 0"* ]]; then
  pass "--spec serial with a draft length refuses: a serial control leg drafts nothing"
else
  fail "serial + draft-len 3 did not refuse (rc ${rc}): ${out}"
fi
out="$("${TREE_UP}" --boot --spec mtp --draft-len 0 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"mtp requires --draft-len between 1 and 7"* ]]; then
  pass "--spec mtp with draft length 0 refuses"
else
  fail "mtp + draft-len 0 did not refuse (rc ${rc}): ${out}"
fi
out="$("${TREE_UP}" --boot --spec sideways --draft-len 1 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--spec must be 'serial', 'mtp' or 'dflash'"* ]]; then
  pass "an unknown --spec value refuses by name"
else
  fail "an unknown --spec did not refuse (rc ${rc}): ${out}"
fi
release_lock

if [[ "${fails}" -eq 0 ]]; then
  printf 'test-resident-up: all cases pass\n'
  exit 0
fi
printf 'test-resident-up: %s case(s) failed\n' "${fails}" >&2
exit 1
