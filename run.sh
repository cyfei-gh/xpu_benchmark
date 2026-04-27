#!/bin/bash
# xpu_benchmark run script
# Usage: bash run.sh [config]

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${1:-"${SCRIPT_DIR}/config/basic.json"}
OUTPUT_DIR="${SCRIPT_DIR}/results"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "xpu_benchmark"
echo "========================================"

python3 get_gpu_spec.py 0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m xpu_benchmark \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}"

torchrun --nproc_per_node=1 -m xpu_benchmark.bench_comm
