#!/usr/bin/env bash
set -euo pipefail

validation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${validation_dir}/../.." && pwd)"
cli="${MMPBSA_CLI:-mmpbsa}"
runtime="${MMPBSA_RUNTIME:-${validation_dir}/runtime.local.yaml}"
workspace="${validation_dir}/_runs/production"
logs="${validation_dir}/_runs/logs"

mkdir -p "${logs}"
cd "${project_root}"

"${cli}" \
  --workflow "${validation_dir}/workflow.yaml" \
  --runtime "${runtime}" \
  --workspace "${workspace}" \
  batch run --through scoring \
  > "${logs}/production.batch.json" \
  2> "${logs}/production.batch.stderr.log"

"${cli}" \
  --workflow "${validation_dir}/workflow.yaml" \
  --runtime "${runtime}" \
  --workspace "${workspace}" \
  finalize case --case lig_ejm_31 \
  > "${logs}/production.finalize-case.json" \
  2> "${logs}/production.finalize-case.stderr.log"

"${cli}" \
  --workflow "${validation_dir}/workflow.yaml" \
  --runtime "${runtime}" \
  --workspace "${workspace}" \
  finalize project \
  > "${logs}/production.finalize-project.json" \
  2> "${logs}/production.finalize-project.stderr.log"
