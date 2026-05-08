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
python -m xpu_benchmark --config config/deepseek.json --output ./results/

# 多卡通信测试（需 torchrun 启动）
torchrun --nproc_per_node=8 -m xpu_benchmark.bench_comm \
    --config ./config/basic.json --output ./results/
```

## 配置文件

配置文件为 JSON 格式，按需包含以下任意 section：`llm_gemm`、`memory`、`comm`。

```json
{
    "llm_gemm": {
        "model": "deepseek-v3",
        "batch_sizes": [1, 4, 16, 64, 256, 1024, 4096],
        "dtypes": ["bfloat16"],
        "tp": 1,
        "num_iters": 30,
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
- `config/deepseek.json` - DeepSeek-V3 LLM GEMM 配置
- `config/basic.json` - 快速验证配置

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
Size(M,N,K)       | Dtype    | Time(ms) | TFLOPS
------------------|----------|----------|--------
1x4096x4096       | bfloat16 | 0.023    | 1456.3
1024x4096x4096    | bfloat16 | 12.8     | 268.4
4096x4096x4096    | bfloat16 | 198.2    | 275.1
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
│   ├── deepseek.json
│   └── basic.json
└── results/            # 结果输出目录
```
