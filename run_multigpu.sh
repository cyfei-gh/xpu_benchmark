#!/bin/bash

RES_DIR="results_multigpu"
mkdir -p ${RES_DIR}

GPU_INDEX_LIST="0 1 2 3 4 5 6 7"

for i in $(seq 1 1000); do
    echo "Running test $i..."
    for GPU_ID in $GPU_INDEX_LIST; do
        # SKIP_COMM=1 跳过 torchrun xpu_benchmark.bench_comm 通信测试，
        # 避免并行多卡单进程场景下的 rendezvous 端口冲突（EADDRINUSE）
        CUDA_VISIBLE_DEVICES=$GPU_ID SKIP_COMM=1 \
            bash run.sh gpu 1 > ${RES_DIR}/gpu_${GPU_ID}.log 2>&1&
    done
    wait
done

wait
echo "All 8 GPU tests completed."

# nvidia-smi -pm 1 # 打开persistence mode
# nvidia-smi -lmc 14001 # 锁频
# nvidia-smi -rmc # 不锁频
