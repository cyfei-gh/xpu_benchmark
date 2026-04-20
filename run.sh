#!/bin/bash
# xpu_benchmark run script
# Usage: bash run.sh [config]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${1:-"${SCRIPT_DIR}/config/basic.json"}
OUTPUT_DIR="${SCRIPT_DIR}/results"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

echo "========================================"
echo "xpu_benchmark"
echo "Config : ${CONFIG}"
echo "Output : ${OUTPUT_DIR}"
echo "PYTHONPATH : ${PYTHONPATH}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

python3 -m xpu_benchmark \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}"
