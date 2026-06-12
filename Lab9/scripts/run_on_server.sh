#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${LAB_DIR}/results"

mkdir -p "${RESULT_DIR}"

make -C "${LAB_DIR}" clean all

"${LAB_DIR}/build/hello_world" 12 4 4 > "${RESULT_DIR}/hello_world.txt"
"${LAB_DIR}/build/matrix_transpose" 512 16 20 > "${RESULT_DIR}/transpose_single_512_16.txt"
"${LAB_DIR}/build/matrix_transpose" --benchmark 20 > "${RESULT_DIR}/transpose_benchmark.csv"

printf 'Results written to %s\n' "${RESULT_DIR}"
