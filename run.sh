#!/bin/bash
# xpu_benchmark run script
# Usage: bash run.sh [mode] [config]
#   mode: all | gemm | membw (default: all)
#   config: path to JSON config (default: config/l20.json)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=${1:-all}
CONFIG=${2:-"${SCRIPT_DIR}/config/l20.json"}
OUTPUT_DIR="${SCRIPT_DIR}/results"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

echo "========================================"
echo "xpu_benchmark"
echo "Mode   : ${MODE}"
echo "Config : ${CONFIG}"
echo "Output : ${OUTPUT_DIR}"
echo "PYTHONPATH : ${PYTHONPATH}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

python3 -m xpu_benchmark \
    --mode "${MODE}" \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}"
