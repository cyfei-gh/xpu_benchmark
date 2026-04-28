#!/usr/bin/env python3
"""
xpu_benchmark main entry point.

Usage:
    # Run benchmarks defined in config file
    python -m xpu_benchmark --config config/deepseek.json

Config file format (JSON):
    Only the sections present in config will be executed.
    Supported sections: "gemm", "memory", "llm_gemm", "comm"

    Example:
    {
        "gemm": {
            "num_iters": 30,
            "dry_run_iters": 5,
            "dtypes": ["float32", "bfloat16"],
            "sizes": [[4096, 4096, 4096], [1, 4096, 4096]]
        },
        "memory": {
            "num_iters": 50,
            "dry_run_iters": 10,
            "dtypes": ["float32"],
            "patterns": ["seq_copy", "seq_read"]
        },
        "llm_gemm": {
            "model": "deepseek-v3",
            "batch_sizes": [1, 4, 16, 64, 256, 1024, 4096],
            "dtypes": ["bfloat16"],
            "tp": 1,
            "num_iters": 30,
            "dry_run_iters": 5
        },
        "comm": {
            "num_iters": 50,
            "dry_run_iters": 10,
            "world_size": [2, 4, 8],
            "operations": ["allreduce", "allgather", "all2all", "all2allv"],
            "dtype": "bfloat16"
        }
    }
"""

import argparse
import json
import os
import sys
import torch
from datetime import datetime

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xpu_benchmark import GemmBenchmark, MemBwBenchmark, LLMGemmBenchmark, CommBenchmark


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_device_prefix() -> str:
    """Get a short device name prefix for output file naming.

    Examples:
        "NVIDIA L20"           -> "L20"
        "NVIDIA H100 80GB HBM3" -> "H100"
        "NVIDIA A100-SXM4-80GB" -> "A100"
    """
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    # Remove vendor prefix
    for vendor in ("NVIDIA ", "AMD ", "Intel "):
        if name.startswith(vendor):
            name = name[len(vendor):]
            break
    # Take the first token, strip common suffixes
    token = name.split()[0] if name else "gpu"
    token = token.split('-')[0]
    # Sanitize for filesystem
    token = ''.join(c if c.isalnum() else '_' for c in token)
    return token or "gpu"


def run_llm_gemm(config: dict, output_dir: str = None, use_cuda_events: bool = False):
    """Run LLM GEMM benchmark (QKV, Proj, FFN, MoE workloads) based on config."""
    cfg = config['llm_gemm']
    model_name = cfg.get('model', 'HY-image-3.0')
    batch_sizes = cfg.get('batch_sizes', [1])
    dtypes = cfg.get('dtypes', ['bfloat16'])
    tp = cfg.get('tp', 1)
    num_iters = cfg.get('num_iters', 30)
    dry_run_iters = cfg.get('dry_run_iters', 5)

    bench = LLMGemmBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_cuda_events,
    )

    results = bench.run(
        model_name=model_name,
        batch_sizes=batch_sizes,
        dtypes=dtypes,
        tp=tp,
    )
    bench.print_summary(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        device_prefix = get_device_prefix()
        csv_path = os.path.join(output_dir, f'{device_prefix}_gemm_{model_name}_{timestamp}.csv')
        bench.save_csv(results, csv_path)
        plot_path = os.path.join(output_dir, f'{device_prefix}_gemm_{model_name}_{timestamp}.png')
        bench.plot_batch_tflops_curve(results, plot_path)

    return results


def run_membw(config: dict, output_dir: str = None, use_cuda_events: bool = False):
    """Run memory bandwidth benchmark based on config."""
    cfg = config['memory']
    sizes_mb = cfg.get('sizes_mb', None)
    patterns = cfg.get('patterns', ['seq_copy', 'strided_copy'])
    dtypes = cfg.get('dtypes', ['float32'])
    num_iters = cfg.get('num_iters', 50)
    dry_run_iters = cfg.get('dry_run_iters', 10)

    bench = MemBwBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_cuda_events,
    )

    results = bench.run(
        sizes_mb=sizes_mb,
        patterns=patterns,
        dtypes=dtypes,
    )
    bench.print_summary(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        device_prefix = get_device_prefix()
        csv_path = os.path.join(output_dir, f'{device_prefix}_membw_{timestamp}.csv')
        bench.save_csv(results, csv_path)
        plot_path = os.path.join(output_dir, f'{device_prefix}_membw_{timestamp}.png')
        bench.plot_size_bw_curve(results, plot_path)

    return results


def run_comm(config: dict, output_dir: str = None):
    """Run communication bandwidth benchmark based on config.

    注意：comm benchmark 需要通过 torchrun 启动多进程环境。
    如果当前未处于分布式环境，会尝试初始化单卡模式。
    """
    cfg = config['comm']
    num_iters = cfg.get('num_iters', 50)
    dry_run_iters = cfg.get('dry_run_iters', 10)
    operations = cfg.get('operations', ['allreduce', 'allgather', 'all2all', 'all2allv'])
    dtype = cfg.get('dtype', 'bfloat16')
    sizes_bytes = cfg.get('sizes_bytes', None)

    # 解析 world_size 参数：支持整数或列表
    world_size_cfg = cfg.get('world_size', None)
    if world_size_cfg is not None:
        if isinstance(world_size_cfg, int):
            world_sizes = [world_size_cfg]
        elif isinstance(world_size_cfg, list):
            world_sizes = world_size_cfg
        else:
            world_sizes = None
    else:
        world_sizes = None

    bench = CommBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
    )

    results = bench.run(
        sizes_bytes=sizes_bytes,
        operations=operations,
        dtype=dtype,
        world_sizes=world_sizes,
    )

    if bench.rank == 0:
        bench.print_summary(results)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            device_prefix = get_device_prefix()
            csv_path = os.path.join(output_dir, f'{device_prefix}_comm_bw_{timestamp}.csv')
            bench.save_csv(results, csv_path)
            plot_path = os.path.join(output_dir, f'{device_prefix}_comm_bw_{timestamp}.png')
            bench.plot(results, plot_path)

    return results


def print_device_info():
    """Print GPU device information."""
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device available.")
        return

    print(f"\n{'='*60}")
    print("GPU Device Information")
    print(f"{'='*60}")
    props = torch.cuda.get_device_properties(0)
    print(f"  Name            : {props.name}")
    print(f"  Total Memory    : {props.total_memory / 1024**3:.1f} GB")
    print(f"  SM Count        : {props.multi_processor_count}")
    print(f"  Compute Cap.    : {props.major}.{props.minor}")
    print(f"  CUDA Version    : {torch.version.cuda}")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="xpu_benchmark: GPU GEMM and Memory Bandwidth Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to JSON config file (required). "
    )
    parser.add_argument(
        '--output',
        type=str,
        default="./results/",
        help="Directory to save CSV/plot results",
    )
    parser.add_argument(
        '--use_cuda_events',
        action='store_true',
        help="Force CUDA Events timing (skip CUPTI even if available)",
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device available. Exiting.")
        sys.exit(1)

    # Print device info
    print_device_info()

    # Load config
    try:
        config = load_config(args.config)
        print(f"[INFO] Loaded config from: {args.config}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

    # Validate config has at least one benchmark section
    valid_sections = ['gemm', 'memory', 'llm_gemm', 'comm']
    found_sections = [s for s in valid_sections if s in config]
    if not found_sections:
        print(f"[ERROR] Config must contain at least one benchmark section: {valid_sections}")
        sys.exit(1)

    print(f"[INFO] Benchmark sections to run: {found_sections}")

    # Run benchmarks based on config sections
    if 'llm_gemm' in config:
        run_llm_gemm(config, output_dir=args.output, use_cuda_events=args.use_cuda_events)

    if 'memory' in config:
        run_membw(config, output_dir=args.output, use_cuda_events=args.use_cuda_events)

    if 'comm' in config:
        run_comm(config, output_dir=args.output)

    print("\n[INFO] Benchmark complete.")


if __name__ == '__main__':
    main()
