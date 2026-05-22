# xpu_benchmark

GPU GEMM、内存带宽与多卡通信性能基准测试工具。

## 功能特性

- **LLM GEMM 基准测试**：基于内置模型配置（QKV / Proj / FFN / MoE）测试不同 batch、dtype、TP 下的矩阵乘法性能
- **内存带宽测试**：测试不同访问模式与数据类型下的 HBM / L2 带宽
- **多卡通信测试**：基于 `torch.distributed` 测试 AllReduce / AllGather / All2All / All2Allv 的有效带宽
- **多种数据类型支持**：float32、bfloat16、float16、int8、float8_e4m3fn
- **高精度计时**：优先使用 CUPTI 硬件级计时，自动回退到 CUDA Events
- **结果保存**：CSV 数据 + 性能曲线 PNG 可视化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：cupti-python 需要 CUDA 13+，如未安装会自动使用 CUDA Events 计时。

### 运行测试

```bash
# 一键运行（默认配置）
bash run.sh gpu 4 | tee bench.log

# 指定配置文件
python -m xpu_benchmark --config config/basic.json --output ./results/

# 多卡通信测试（需 torchrun 启动）
torchrun --nproc_per_node=8 -m xpu_benchmark.bench_comm \
    --config ./config/basic.json --output ./results/
```

## 配置文件

配置文件为 JSON 格式，按需包含以下任意 section：`llm_gemm`、`memory`、`comm`。

```json
{
    "llm_gemm": {
        "model": "Basic",
        "batch_sizes": [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 4099, 5000, 8192],
        "dtypes": ["fp32", "bf16", "fp16", "int8", "fp8_tensorwise", "fp8_rowwise", "mxfp8", "nvfp4"],
        "tp": 1,
        "num_iters": 10,
        "dry_run_iters": 5
    },
    "memory": {
        "num_iters": 50,
        "dry_run_iters": 10,
        "dtypes": ["float32"],
        "patterns": ["seq_copy", "seq_read", "strided_copy"],
        "flush_l2_cache": false
    },
    "comm": {
        "num_iters": 50,
        "dry_run_iters": 10,
        "world_sizes": [2, 4, 8],
        "operations": ["allreduce", "allgather", "all2all", "all2allv"],
        "dtype": "bfloat16"
    }
}
```

预置配置文件：
- `config/basic.json` - 全面测试配置

## 输出结果

测试结果保存至 `--output` 指定目录（默认 `./results/`），文件名统一带 `{device_prefix}` 前缀（如 `H20`、`L20`）以及时间戳，方便跨设备对比：

| Benchmark | 文件名格式 |
|-----------|-----------|
| LLM GEMM  | `{device}_gemm_{model}_{timestamp}.csv` / `.png` |
| 内存带宽  | `{device}_membw_{HBM\|L2}_{timestamp}.csv` / `.png` |
| 多卡通信  | `TP{ws}_{device}_comm_bw_{timestamp}.png`，CSV 为 `{device}_comm_bw_{timestamp}.csv` |

> 内存带宽中 `HBM` 表示开启 `flush_l2_cache`，`L2` 表示保留 L2 命中。

### GEMM 结果示例

```
--- dtype: bf16 ---
workload          batch |      M      N      K |  torch                          |  tilelang                       |  triton                         |
                        |                      |    time(ms)     TFLOPS      MFU |    time(ms)     TFLOPS      MFU |    time(ms)     TFLOPS      MFU |                
------------------------------------------------------------------------------------------------------------------------------------------------------
Mx4096x4096           1 |      1   4096   4096 |       0.083       0.40    30.0% |       0.052       0.64    48.0% |       0.467       0.07     5.3% |
Mx4096x4096           2 |      2   4096   4096 |       0.058       1.15    42.8% |       0.052       1.29    48.0% |       0.492       0.14     5.1% |
Mx4096x4096           3 |      3   4096   4096 |       0.059       1.72    42.6% |       0.051       1.96    48.7% |       0.491       0.21     5.1% |
Mx4096x4096           4 |      4   4096   4096 |       0.059       2.27    42.2% |       0.052       2.59    48.3% |       0.493       0.27     5.1% |
Mx4096x4096           8 |      8   4096   4096 |       0.061       4.43    41.3% |       0.050       5.37    50.1% |       0.490       0.55     5.1% |
Mx4096x4096          16 |     16   4096   4096 |       0.046      11.58    54.3% |       0.044      12.21    57.2% |       0.491       1.09     5.1% |
Mx4096x4096          32 |     32   4096   4096 |       0.072      15.01    35.5% |       0.044      24.13    57.0% |       0.497       2.16     5.1% |
Mx4096x4096          64 |     64   4096   4096 |       0.050      42.87    51.4% |       0.048      45.16    54.1% |       0.506       4.24     5.1% |
Mx4096x4096         128 |    128   4096   4096 |       0.052      82.17    50.7% |       0.063      68.60    42.4% |       0.513       8.37     5.2% |
Mx4096x4096         256 |    256   4096   4096 |       0.079     108.85    43.5% |       0.086     100.09    40.0% |       0.500      17.18     6.9% |
Mx4096x4096        1024 |   1024   4096   4096 |       0.204     168.07    67.2% |       0.199     172.49    69.0% |       0.765      44.93    18.0% |
Mx4096x4096        2050 |   2050   4096   4096 |       0.353     194.80    77.9% |       0.370     185.96    74.4% |       1.554      44.27    17.7% |
Mx4096x4096        4096 |   4096   4096   4096 |       0.585     235.06    94.0% |       0.632     217.39    87.0% |       2.383      57.68    23.1% |
Mx4096x4096        4099 |   4099   4096   4096 |       0.587     234.24    93.7% |       0.634     217.08    86.8% |       2.398      57.34    22.9% |
Mx4096x4096        5000 |   5000   4096   4096 |       0.703     238.61    95.4% |       0.755     222.17    88.9% |       2.813      59.64    23.9% |
Mx4096x4096        8192 |   8192   4096   4096 |       1.170     234.87    93.9% |       1.191     230.83    92.3% |       4.476      61.41    24.6% |

```

## 目录结构

```
xpu_benchmark/
├── run.sh              # 运行脚本
├── __main__.py         # 主入口（设备信息 + 调度各 benchmark）
├── bench_gemm.py       # LLM GEMM 基准测试
├── bench_memory.py     # 内存带宽基准测试
├── bench_comm.py       # 多卡通信基准测试
├── timing.py           # CUPTI / CUDA Events 计时工具
├── hw_spec.py          # 硬件规格与 device_prefix
├── xpu_device.py       # CUDA / NPU 抽象
├── config/             # 配置文件目录
│   └── basic.json
└── results/            # 结果输出目录
```
