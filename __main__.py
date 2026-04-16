#!/usr/bin/env python3
"""
xpu_benchmark main entry point.

Usage:
    # Run benchmarks defined in config file
    python -m xpu_benchmark --config config/deepseek.json

    # Run and save results to output directory
    python -m xpu_benchmark --config config/deepseek.json --output results/

    # Use CUDA Events timing instead of CUPTI
    python -m xpu_benchmark --config config/deepseek.json --use_cuda_events

Config file format (JSON):
    Only the sections present in config will be executed.
    Supported sections: "gemm", "membw", "llm_gemm"

    Example:
    {
        "gemm": {
            "num_iters": 30,
            "dry_run_iters": 5,
            "dtypes": ["float32", "bfloat16"],
            "sizes": [[4096, 4096, 4096], [1, 4096, 4096]]
        },
        "membw": {
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

from xpu_benchmark import GemmBenchmark, MemBwBenchmark, LLMGemmBenchmark, LLM_MODELS


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_gemm(config: dict, output_dir: str = None, use_cuda_events: bool = False):
    """Run GEMM benchmark based on config."""
    cfg = config['gemm']
    sizes = [tuple(s) for s in cfg.get('sizes', [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ])]
    dtypes = cfg.get('dtypes', ['float32', 'bfloat16', 'float16'])
    num_iters = cfg.get('num_iters', 30)
    dry_run_iters = cfg.get('dry_run_iters', 5)

    bench = GemmBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_cuda_events,
    )

    results = bench.run(sizes=sizes, dtypes=dtypes)
    bench.print_summary(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f'gemm_{timestamp}.csv')
        bench.save_csv(results, csv_path)

    return results


def run_membw(config: dict, output_dir: str = None, use_cuda_events: bool = False):
    """Run memory bandwidth benchmark based on config."""
    cfg = config['membw']
    sizes_mb = cfg.get('sizes_mb', None)
    patterns = cfg.get('patterns', ['seq_copy', 'strided_copy'])
    dtypes = cfg.get('dtypes', ['float32'])
    num_iters = cfg.get('num_iters', 50)
    dry_run_iters = cfg.get('dry_run_iters', 10)
    block_counts = cfg.get('block_counts', None)

    bench = MemBwBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_cuda_events,
    )

    results = bench.run(
        sizes_mb=sizes_mb,
        patterns=patterns,
        dtypes=dtypes,
        block_counts=block_counts,
    )
    bench.print_summary(results)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f'membw_{timestamp}.csv')
        bench.save_csv(results, csv_path)
        plot_path = os.path.join(output_dir, f'membw_heatmap_{timestamp}.png')
        bench.plot(results, plot_path)

    return results


def run_llm_gemm(config: dict, output_dir: str = None, use_cuda_events: bool = False):
    """Run LLM GEMM benchmark (QKV, Proj, FFN, MoE workloads) based on config."""
    cfg = config['llm_gemm']
    model_name = cfg.get('model', 'deepseek-v3')
    batch_sizes = cfg.get('batch_sizes', None)
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
        csv_path = os.path.join(output_dir, f'llm_gemm_{model_name}_{timestamp}.csv')
        bench.save_csv(results, csv_path)
        plot_path = os.path.join(output_dir, f'llm_gemm_{model_name}_{timestamp}.png')
        bench.plot_batch_tflops(results, plot_path)

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
    valid_sections = ['gemm', 'membw', 'llm_gemm']
    found_sections = [s for s in valid_sections if s in config]
    if not found_sections:
        print(f"[ERROR] Config must contain at least one benchmark section: {valid_sections}")
        sys.exit(1)

    print(f"[INFO] Benchmark sections to run: {found_sections}")

    # Run benchmarks based on config sections
    if 'gemm' in config:
        run_gemm(config, output_dir=args.output, use_cuda_events=args.use_cuda_events)

    if 'membw' in config:
        run_membw(config, output_dir=args.output, use_cuda_events=args.use_cuda_events)

    if 'llm_gemm' in config:
        run_llm_gemm(config, output_dir=args.output, use_cuda_events=args.use_cuda_events)

    print("\n[INFO] Benchmark complete.")


if __name__ == '__main__':
    main()
