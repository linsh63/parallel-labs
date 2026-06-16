#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${LAB_DIR}/results"

mkdir -p "${RESULT_DIR}"

make -C "${LAB_DIR}" clean all

"${LAB_DIR}/build/matrix_multiply" 512 512 512 16 5 \
    > "${RESULT_DIR}/matmul_single_512_16.txt"
"${LAB_DIR}/build/matrix_multiply" --benchmark 1 \
    > "${RESULT_DIR}/matmul_benchmark.csv"

printf 'Results written to %s\n' "${RESULT_DIR}"
