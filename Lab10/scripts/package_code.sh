#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="${LAB_DIR}/Lab10_code.zip"

cd "${LAB_DIR}"
rm -f "${PACKAGE}"
zip -r "${PACKAGE}" Makefile README.md src scripts -x '*/.DS_Store'

printf 'Created %s\n' "${PACKAGE}"
