#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${LAB_DIR}/results"

mkdir -p "${RESULT_DIR}"

if [[ -z "${NVCC:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [[ -x "${candidate}" ]]; then
            export NVCC="${candidate}"
            export PATH="$(dirname "${candidate}"):${PATH}"
            break
        fi
    done
fi

if [[ -n "${NVCC:-}" ]]; then
    printf 'Using NVCC=%s\n' "${NVCC}"
elif command -v nvcc >/dev/null 2>&1; then
    printf 'Using NVCC=%s\n' "$(command -v nvcc)"
else
    printf 'Error: nvcc not found. Load CUDA or set NVCC=/path/to/nvcc.\n' >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && ! nvidia-smi -L >/dev/null 2>&1; then
    printf 'Error: NVIDIA driver/GPU is not available in this shell. Run this script on a GPU node or fix the driver environment.\n' >&2
    exit 1
fi

make -C "${LAB_DIR}" clean all

"${LAB_DIR}/build/matrix_multiply" 512 512 512 16 5 \
    > "${RESULT_DIR}/matmul_single_512_16.txt"
"${LAB_DIR}/build/matrix_multiply" --benchmark 1 \
    > "${RESULT_DIR}/matmul_benchmark.csv"

printf 'Results written to %s\n' "${RESULT_DIR}"
