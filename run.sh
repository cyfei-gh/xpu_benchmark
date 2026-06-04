#!/bin/bash
# xpu_benchmark run script
# Usage:
#   bash run.sh <backend> <nproc> [config]
#
# 参数说明：
#   backend : 设备类型，可选值 cpu | gpu | npu
#             （gpu 等同于 cuda，使用 NCCL；npu 使用 HCCL；cpu 使用 Gloo）
#   nproc   : 通信 benchmark 参与的进程/卡数（torchrun --nproc_per_node）
#   config  : 可选，benchmark 配置文件路径，默认 config/basic.json
#   优先使用外部环境变量 CUDA_VISIBLE_DEVICES，否则使用 default_devices_list
#
# 示例：
#   bash run.sh gpu 8 ./config/basic.json | tee "bench.log"
#   bash run.sh npu 4 ./config/basic.json
#   bash run.sh cpu 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

# ---------------------------------------------------------------------
# 解析命令行参数
# ---------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: bash run.sh <backend: cpu|gpu|npu> <nproc> [config]" >&2
    exit 1
fi

BACKEND_INPUT="$1"
NPROC="$2"
CONFIG="${3:-${SCRIPT_DIR}/config/basic.json}"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# 归一化 backend：gpu -> cuda
BACKEND="$(echo "${BACKEND_INPUT}" | tr '[:upper:]' '[:lower:]')"
case "${BACKEND}" in
    gpu|cuda)
        BACKEND="cuda"
        ;;
    npu)
        BACKEND="npu"
        ;;
    cpu)
        BACKEND="cpu"
        ;;
    *)
        echo "[ERROR] 不支持的 backend: ${BACKEND_INPUT}（仅支持 cpu | gpu | npu）" >&2
        exit 1
        ;;
esac

# 校验 nproc 为正整数
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] nproc 必须为正整数，收到: ${NPROC}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "xpu_benchmark  (backend = ${BACKEND})"
echo "========================================"

# ---------------------------------------------------------------------
# 优先使用外部环境变量 CUDA_VISIBLE_DEVICES，否则使用 default_devices_list
# ---------------------------------------------------------------------
if [[ "${BACKEND}" == "cuda" ]]; then
    # 生成 default_devices_list = {0, 1, 2, ... , NPROC}
    DEFAULT_CUDA_DEVICES=""
    for ((i=0; i<NPROC; i++)); do
        if [[ -z "${DEFAULT_CUDA_DEVICES}" ]]; then
            DEFAULT_CUDA_DEVICES="${i}" # 如果是第一个设备，直接赋值
        else
            DEFAULT_CUDA_DEVICES="${DEFAULT_CUDA_DEVICES},${i}" # 后续设备用逗号分隔
        fi
    done
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}
    VISIBLE_LIST=${CUDA_VISIBLE_DEVICES}
elif [[ "${BACKEND}" == "npu" ]]; then
    DEFAULT_NPU_DEVICES=""
    for ((i=0; i<NPROC; i++)); do
        if [[ -z "${DEFAULT_NPU_DEVICES}" ]]; then
            DEFAULT_NPU_DEVICES="${i}"
        else
            DEFAULT_NPU_DEVICES="${DEFAULT_NPU_DEVICES},${i}"
        fi
    done
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${DEFAULT_NPU_DEVICES}}
    export NPU_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}
    VISIBLE_LIST=${ASCEND_RT_VISIBLE_DEVICES}
else
    # cpu 后端无可见设备概念
    VISIBLE_LIST="N/A (cpu backend, using gloo)"
fi

# 若为 cuda/npu，校验 nproc 不超过可见设备数
if [[ "${BACKEND}" == "cuda" || "${BACKEND}" == "npu" ]]; then
    NUM_VISIBLE=$(echo "${VISIBLE_LIST}" | awk -F',' '{print NF}')
    if (( NPROC > NUM_VISIBLE )); then
        echo "[ERROR] nproc(${NPROC}) 超过可见设备数(${NUM_VISIBLE}): ${VISIBLE_LIST}" >&2
        exit 1
    fi
fi

echo "Visible devices : {${VISIBLE_LIST}}"
echo "nproc_per_node  : ${NPROC}"
echo "config          : ${CONFIG}"
echo "----------------------------------------"

# ---------------------------------------------------------------------
# 运行 GEMM / MemBw / LLM_GEMM / Comm benchmark
# （设备硬件信息由 xpu_benchmark.__main__.print_device_info 打印）
# ---------------------------------------------------------------------
python3 -m xpu_benchmark --config "${CONFIG}" --output "${OUTPUT_DIR}"

# ---------------------------------------------------------------------
# 运行通信 benchmark（多进程，后端自动匹配 nccl / hccl / gloo）
# ---------------------------------------------------------------------
# 若设置 SKIP_COMM 则跳过通信 benchmark（例如并行跑多卡单进程时，
# 多个 torchrun 会争抢 rendezvous 端口导致 EADDRINUSE 冲突）
if [[ -n "${SKIP_COMM:-}" ]]; then
    echo "[INFO] 已设置 SKIP_COMM, 跳过通信 benchmark (torchrun xpu_benchmark.bench_comm)"
else
    torchrun --nproc_per_node=${NPROC} -m xpu_benchmark.bench_comm \
        --config "${CONFIG}" --output "${OUTPUT_DIR}"
fi
