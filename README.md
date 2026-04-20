# xpu_benchmark

GPU GEMM 和内存带宽性能基准测试工具。

## 功能特性

- **GEMM 基准测试**：测试不同矩阵尺寸和数据类型的矩阵乘法性能
- **内存带宽测试**：测试不同访问模式下的内存带宽性能
- **多种数据类型支持**：float32、bfloat16、float16、int8、float8_e4m3fn
- **高精度计时**：优先使用 CUPTI 硬件级计时，自动回退到 CUDA Events
- **结果保存**：支持 CSV 格式输出和热力图可视化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：cupti-python 需要 CUDA 13+，如未安装会自动使用 CUDA Events 计时。

### 运行测试

使用默认配置运行所有测试：

```bash
bash run.sh

# 指定配置文件
python -m xpu_benchmark --config config/deepseek.json
```

## 配置文件

配置文件为 JSON 格式，包含 GEMM 和内存带宽测试参数：

```json
{
    "gemm": {
        "num_iters": 30,
        "dry_run_iters": 5,
        "dtypes": ["float32", "bfloat16", "float16"],
        "sizes": [
            [1, 4096, 4096],
            [1024, 4096, 4096],
            [4096, 4096, 4096]
        ]
    },
    "memory": {
        "num_iters": 50,
        "dry_run_iters": 10,
        "dtypes": ["float32"],
        "patterns": ["seq_copy", "seq_read", "strided_copy"]
    }
}
```

预置配置文件：
- `config/deepseek.json` - NVIDIA L20 GPU 配置
- `config/basic.json` - 快速测试配置

## 输出结果

测试结果保存至 `results/` 目录：

- `gemm_YYYYMMDD_HHMMSS.csv` - GEMM 性能数据
- `membw_YYYYMMDD_HHMMSS.csv` - 内存带宽数据
- `membw_heatmap_YYYYMMDD_HHMMSS.png` - 带宽热力图

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
├── __main__.py         # 主入口
├── bench_gemm.py       # GEMM 基准测试
├── bench_memory.py     # 内存带宽基准测试
├── timing.py           # 计时工具
├── hw_spec.py          # 硬件规格
├── config/             # 配置文件目录
│   ├── deepseek.json
│   └── basic.json
└── results/            # 结果输出目录
```
