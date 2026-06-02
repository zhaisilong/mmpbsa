#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/pipeline_tests/peptide_3x5ns}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/configs/peptide_crystal_3x5ns.yaml}"
PYTHON="${PYTHON:-python}"
NTOMP="${NTOMP:-4}"
MMPBSA_NP="${MMPBSA_NP:-16}"

mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/gpu4.pid "$RUN_DIR"/gpu5.pid "$RUN_DIR"/gpu6.pid "$RUN_DIR"/gpu7.pid

run_group() {
  local gpu="$1"
  shift
  (
    cd "$PROJECT_ROOT"
    export GPU_ID="$gpu" NTOMP MMPBSA_NP PYTHONUNBUFFERED=1
    for job in "$@"; do
      date -u "+%Y-%m-%dT%H:%M:%SZ start ${job} GPU=${gpu}"
      printf "command: GPU_ID=%s NTOMP=%s MMPBSA_NP=%s %s -m mmpbsa peptide run %s --job-id %s --protocol %s --resume\n" \
        "$GPU_ID" "$NTOMP" "$MMPBSA_NP" "$PYTHON" "$RUN_DIR" "$job" "$PROTOCOL"
      set +e
      "$PYTHON" -m mmpbsa peptide run "$RUN_DIR" \
        --job-id "$job" --protocol "$PROTOCOL" --resume
      rc=$?
      set -e
      printf "exit_code=%s job=%s GPU=%s\n" "$rc" "$job" "$gpu"
      if [[ "$rc" -ne 0 ]]; then
        exit "$rc"
      fi
      date -u "+%Y-%m-%dT%H:%M:%SZ done ${job} GPU=${gpu}"
    done
  ) > "$RUN_DIR/gpu${gpu}.log" 2>&1 &
  echo $! > "$RUN_DIR/gpu${gpu}.pid"
}

run_group 4 sp2016_09 sp2016_01 sp2016_14
run_group 5 sp2016_17 sp2016_02 sp2016_16
run_group 6 sp2016_15 sp2016_05 sp2016_10
run_group 7 sp2016_03 sp2016_08 sp2016_06

for gpu in 4 5 6 7; do
  printf "gpu%s_pid=%s log=%s\n" "$gpu" "$(cat "$RUN_DIR/gpu${gpu}.pid")" "$RUN_DIR/gpu${gpu}.log"
done

wait
