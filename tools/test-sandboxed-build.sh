#!/usr/bin/env bash
#
# test-sandboxed-build.sh -- the compile steps of the build run under Seatbelt.
#
# WHAT IS UNDER TEST.
#   tools/sandboxed-build.sh   the trusted wrapper that starts a compile
#                              command under sandbox-exec on an official run
#   tools/seatbelt-profile.py  the build profile (--build)
#
# A compiler reads every file that a source names. The editable surface holds
# C++ and Metal sources, so a source can name a golden or the runner
# registration (#include, #embed, .incbin). Part 1 is hermetic and runs on any
# host: the build profile rules and the refusals of the wrapper. Part 2 needs
# macOS (/usr/bin/sandbox-exec). It runs small commands and one real compile
# with canary evaluator paths in a temporary directory. No engine build, no
# weights, no GPU. The network case uses a listener on 127.0.0.1.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WRAPPER="${REPO_ROOT}/tools/sandboxed-build.sh"
GENERATOR="${REPO_ROOT}/tools/seatbelt-profile.py"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
LISTENER_PID=""
cleanup() {
  [[ -z "${LISTENER_PID}" ]] || kill "${LISTENER_PID}" 2>/dev/null
  chmod -R u+w "${WORK}" 2>/dev/null
  rm -rf "${WORK}"
}
trap cleanup EXIT

passes=0
failures=0
pass() { passes=$((passes + 1)); echo "ok    $*"; }
fail() { failures=$((failures + 1)); echo "FAIL  $*" >&2; }

# The canary evaluator paths. Every one exists, so a deny is observable. Each
# file holds a marker, so a read that succeeds is observable too.
MARKER="CANARY-MARKER-7c1e"
C="${WORK}/canary"
mkdir -p "${C}/goldens" "${C}/benchd-bin" "${C}/baseline" "${C}/private" "${C}/cache" \
  "${C}/runner/_work/repo" "${WORK}/out" "${WORK}/home"
printf '{"canary":"%s"}\n' "${MARKER}" > "${C}/goldens/canary.golden.json"
printf '%s\n' "${MARKER}" > "${C}/benchd-bin/benchd"
printf '%s\n' "${MARKER}" > "${C}/baseline/README.md"
printf '{"canary":"%s"}\n' "${MARKER}" > "${C}/calibration.json"
printf '%s\n' "${MARKER}" > "${C}/private/note"
printf '%s\n' "${MARKER}" > "${C}/cache/product"
printf '{"canary":"%s"}\n' "${MARKER}" > "${C}/runner/.credentials"
printf '{"canary":"%s"}\n' "${MARKER}" > "${C}/runner/.runner"
canary_env=(
  MLXFAST_QWEN38_GOLDEN_DIR="${C}/goldens"
  MLXFAST_BASELINE_WORKSPACE="${C}/baseline"
  MLXFAST_BASELINE_CALIBRATION="${C}/calibration.json"
  BENCHD_BIN_DIR="${C}/benchd-bin"
  MLXFAST_PRIVATE_DIR="${C}/private"
  MLXFAST_BUILD_CACHE_DIR="${C}/cache"
  RUNNER_WORKSPACE="${C}/runner/_work/repo"
)
base_env=(PATH="${PATH}" HOME="${WORK}/home" TMPDIR="${WORK}/")
[[ -z "${DEVELOPER_DIR:-}" ]] || base_env+=(DEVELOPER_DIR="${DEVELOPER_DIR}")
official=(env -i "${base_env[@]}" "${canary_env[@]}" MLXFAST_OFFICIAL_BENCHMARK_RUN=1)

# --- part 1: the build profile and the refusals --------------------------------
profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --build --official --tree "${WORK}/tree" --write-subpath "${WORK}/out")"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  fail "generator: exit ${rc} for a build profile with every evaluator path set"
else
  ok=1
  for path in "${C}/goldens" "${C}/baseline" "${C}/calibration.json" "${C}/benchd-bin" "${C}/private" \
    "${C}/cache" "${C}/runner/.runner" "${C}/runner/.credentials" "${C}/runner/.credentials_rsaparams"; do
    grep -qF "(deny file-read* file-write* (subpath \"${path}\"))" <<< "${profile}" \
      || { ok=0; fail "generator: the build profile has no read and write deny for ${path}"; }
  done
  for rule in "(deny network*)" "(deny file-write*)"; do
    grep -qF "${rule}" <<< "${profile}" || { ok=0; fail "generator: the build profile has no ${rule}"; }
  done
  for rule in "(deny process-fork)" "(deny process-exec*)"; do
    ! grep -qF "${rule}" <<< "${profile}" || { ok=0; fail "generator: the build profile has ${rule}; a build starts compilers"; }
  done
  [[ "$(tail -n 1 <<< "${profile}")" == "(deny file-read* file-write* "* ]] \
    || { ok=0; fail "generator: the evaluator denies are not the last rules of the build profile"; }
  [[ "${ok}" -eq 0 ]] || pass "generator: the build profile denies network and all writes, allows fork and exec, and denies every evaluator path last"
fi

for name in MLXFAST_QWEN38_GOLDEN_DIR MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION BENCHD_BIN_DIR RUNNER_WORKSPACE; do
  out="$("${official[@]}" "${name}=" "${WRAPPER}" --write "${WORK}/out" -- /bin/sh -c 'echo COMMAND-RAN' 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"${name}"* && "${out}" != *COMMAND-RAN* ]]; then
    pass "wrapper: an official build with ${name} unset refuses by name and starts nothing"
  else
    fail "wrapper: an official build with ${name} unset did not refuse (rc ${rc}): ${out}"
  fi
done

out="$("${official[@]}" "${WRAPPER}" -- /bin/sh -c 'echo COMMAND-RAN' 2>&1)"
rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--write"* && "${out}" != *COMMAND-RAN* ]]; then
  pass "wrapper: a call with no --write directory refuses"
else
  fail "wrapper: a call with no --write directory did not refuse (rc ${rc}): ${out}"
fi

out="$(env -i "${base_env[@]}" "${WRAPPER}" --write "${WORK}/out" -- /bin/cat "${C}/goldens/canary.golden.json" 2>&1)"
rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"${MARKER}"* ]]; then
  pass "wrapper: a run that is not official runs the command with no sandbox"
else
  fail "wrapper: a run that is not official did not run the command as it is (rc ${rc}): ${out}"
fi

# --- part 2: the real Seatbelt -------------------------------------------------
if [[ ! -x /usr/bin/sandbox-exec ]]; then
  echo "test-sandboxed-build.sh: part 2 NOT RUN: /usr/bin/sandbox-exec is not on this host"
  echo "test-sandboxed-build.sh: ${passes} passed, ${failures} failed"
  [[ "${failures}" -eq 0 ]]
  exit $?
fi

sandboxed() { "${official[@]}" "${WRAPPER}" --write "${WORK}/out" -- "$@"; }
# A case that expects a failure proves nothing when the wrapper did not start
# the command. The wrapper prints this line when it starts the sandbox.
STARTED="runs under sandbox-exec"

for path in "${C}/goldens/canary.golden.json" "${C}/calibration.json" "${C}/benchd-bin/benchd" \
  "${C}/baseline/README.md" "${C}/private/note" "${C}/cache/product" \
  "${C}/runner/.credentials" "${C}/runner/.runner"; do
  out="$(sandboxed /bin/cat "${path}" 2>&1)"
  rc=$?
  if [[ "${out}" == *"${STARTED}"* && "${rc}" -ne 0 && "${out}" != *"${MARKER}"* && "${out}" == *"Operation not permitted"* ]]; then
    pass "sandbox: a build command cannot read ${path#"${C}/"}"
  else
    fail "sandbox: a build command read ${path} (rc ${rc}): ${out}"
  fi
done

out="$(sandboxed /bin/sh -c 'echo x > "$1/inside" && echo wrote' sh "${WORK}/out" 2>&1)"
if [[ "${out}" == *wrote* && -f "${WORK}/out/inside" ]]; then
  pass "sandbox: a build command writes inside its --write directory"
else
  fail "sandbox: a build command cannot write inside its --write directory: ${out}"
fi

# The temporary directory of the account is writable for a build, so the
# probe is in /private/tmp, which is not.
OUTSIDE="/private/tmp/test-sandboxed-build.$$.outside"
out="$(sandboxed /bin/sh -c 'echo x > "$1" && echo wrote' sh "${OUTSIDE}" 2>&1)"
if [[ "${out}" == *"${STARTED}"* && "${out}" != *wrote* && ! -e "${OUTSIDE}" ]]; then
  pass "sandbox: a build command cannot write outside its --write directory"
else
  fail "sandbox: a build command wrote outside its --write directory: ${out}"
fi
rm -f "${OUTSIDE}"

# /bin/sh writes the temporary file of a here document in /var/tmp.
out="$(sandboxed /bin/sh -c 'cat > "$1/heredoc" <<EOF
x
EOF
echo "rc=$?"' sh "${WORK}/out" 2>&1)"
if [[ "${out}" == *"rc=0"* && -s "${WORK}/out/heredoc" ]]; then
  pass "sandbox: a build script can use a here document"
else
  fail "sandbox: a here document fails in the build sandbox: ${out}"
fi

# The network. A listener on 127.0.0.1 accepts a connection from outside the
# sandbox, so a refusal inside the sandbox is the sandbox.
port_file="${WORK}/port"
python3 - "${port_file}" <<'PYEOF' &
import socket, sys
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(8)
with open(sys.argv[1], "w") as out:
    out.write(str(server.getsockname()[1]))
while True:
    connection, _ = server.accept()
    connection.close()
PYEOF
LISTENER_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -s "${port_file}" ]] && break; sleep 0.3; done
port="$(cat "${port_file}" 2>/dev/null || true)"
connect='import socket, sys
s = socket.socket(); s.settimeout(3)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError as error:
    print("refused:", error); sys.exit(1)
print("connected")'
if [[ -z "${port}" ]]; then
  fail "network: the test listener did not start"
elif [[ "$(python3 -c "${connect}" "${port}" 2>&1)" != connected ]]; then
  fail "network: the control connection from outside the sandbox failed"
else
  out="$(sandboxed "$(command -v python3)" -c "${connect}" "${port}" 2>&1)"
  if [[ "${out}" == *"${STARTED}"* && "${out}" == *refused* ]]; then
    pass "sandbox: a build command cannot use the network"
  else
    fail "sandbox: a build command connected to the network: ${out}"
  fi
fi

# One real compile. The three ways in which a source can name a file.
if ! compiler="$(xcrun --find clang++ 2>/dev/null)" || [[ -z "${compiler}" ]]; then
  echo "test-sandboxed-build.sh: the compile cases are NOT RUN: xcrun finds no clang++"
else
  mkdir -p "${WORK}/src"
  printf '#include "%s"\nint main() { return 0; }\n' "${C}/goldens/canary.golden.json" > "${WORK}/src/include.cpp"
  printf '__asm__(".section __DATA,__const\\n.globl _canary\\n_canary:\\n.incbin \\"%s\\"\\n");\nint main() { return 0; }\n' \
    "${C}/runner/.credentials" > "${WORK}/src/incbin.cpp"
  printf 'int main() { return 0; }\n' > "${WORK}/src/clean.cpp"
  for kind in include incbin; do
    # The control: outside the sandbox the compiler reads the file.
    control="$(env -i "${base_env[@]}" "${compiler}" -std=c++17 -c "${WORK}/src/${kind}.cpp" -o "${WORK}/out/${kind}.control.o" 2>&1)"
    if [[ "${control}" != *"${MARKER}"* ]] && ! grep -qa "${MARKER}" "${WORK}/out/${kind}.control.o" 2>/dev/null; then
      fail "compile (${kind}): the control compile outside the sandbox did not read the canary, so the case proves nothing"
      continue
    fi
    rm -f "${WORK}/out/${kind}.o"
    out="$(sandboxed "${compiler}" -std=c++17 -c "${WORK}/src/${kind}.cpp" -o "${WORK}/out/${kind}.o" 2>&1)"
    rc=$?
    if [[ "${out}" == *"${STARTED}"* && "${rc}" -ne 0 && "${out}" != *"${MARKER}"* ]] \
        && ! grep -qa "${MARKER}" "${WORK}/out/${kind}.o" 2>/dev/null; then
      pass "sandbox: a compile with ${kind} of an evaluator file fails, and the file content is not in the output or the object"
    else
      fail "sandbox: a compile with ${kind} of an evaluator file got the content (rc ${rc}): ${out:0:300}"
    fi
  done
  out="$(sandboxed "${compiler}" -std=c++17 -c "${WORK}/src/clean.cpp" -o "${WORK}/out/clean.o" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 0 && -s "${WORK}/out/clean.o" ]]; then
    pass "sandbox: a compile of a clean source passes"
  else
    fail "sandbox: a compile of a clean source fails in the sandbox (rc ${rc}): ${out:0:300}"
  fi
fi

echo "test-sandboxed-build.sh: ${passes} passed, ${failures} failed"
[[ "${failures}" -eq 0 ]]
