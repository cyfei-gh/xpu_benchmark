#!/usr/bin/env python3
"""
xpu_benchmark main entry point.

Usage:
    # Run all benchmarks with default settings
    python -m xpu_benchmark

    # Run GEMM only
    python -m xpu_benchmark --mode gemm

    # Run memory bandwidth only
    python -m xpu_benchmark --mode membw

    # Run with config file
    python -m xpu_benchmark --config config/l20.json

    # Run with custom sizes
    python -m xpu_benchmark --mode gemm --sizes 4096,4096,4096 64,4096,4096

    # Save results to CSV
    python -m xpu_benchmark --output results/
"""

import argparse
import json
import os
import sys
import torch
from datetime import datetime

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xpu_benchmark import GemmBenchmark, MemBwBenchmark


def parse_size(s: str):
    """Parse 'M,N,K' string into (M, N, K) tuple."""
    parts = s.strip().split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid size '{s}'. Expected format: M,N,K (e.g. 4096,4096,4096)"
        )
    return tuple(int(p) for p in parts)


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_gemm(args, config: dict = None):
    """Run GEMM benchmark."""
    # Determine settings
    if config and 'gemm' in config:
        cfg = config['gemm']
        sizes = [tuple(s) for s in cfg.get('sizes', [])]
        dtypes = cfg.get('dtypes', ['float32', 'bfloat16', 'float16'])
        num_iters = cfg.get('num_iters', 30)
        dry_run_iters = cfg.get('dry_run_iters', 5)
    else:
        # Default sizes covering common LLM workloads
        sizes = [
            # Square matrices
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            # Decode (small M, large N/K) - typical LLM inference
            (1, 4096, 4096),
            (4, 4096, 4096),
            (16, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            # Prefill (large M)
            (1024, 4096, 4096),
            (4096, 4096, 4096),
            # Rectangular
            (64, 8192, 1024),
            (4096, 8192, 1024),
        ]
        dtypes = getattr(args, 'dtypes', None) or ['float32', 'bfloat16', 'float16']
        num_iters = getattr(args, 'num_iters', 30)
        dry_run_iters = getattr(args, 'dry_run_iters', 5)

    # Override with CLI args if provided
    if hasattr(args, 'sizes') and args.sizes:
        sizes = list(args.sizes)
    if hasattr(args, 'dtypes') and args.dtypes:
        dtypes = args.dtypes

    bench = GemmBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not getattr(args, 'use_cuda_events', False),
    )

    results = bench.run(sizes=sizes, dtypes=dtypes)
    bench.print_summary(results)

    if getattr(args, 'output', None):
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(args.output, f'gemm_{timestamp}.csv')
        bench.save_csv(results, csv_path)

    return results


def run_membw(args, config: dict = None):
    """Run memory bandwidth benchmark."""
    if config and 'membw' in config:
        cfg = config['membw']
        sizes_mb = cfg.get('sizes_mb', None)  # None = auto-generate around L2
        patterns = cfg.get('patterns', ['seq_copy', 'strided_copy'])
        dtypes = cfg.get('dtypes', ['float32'])
        num_iters = cfg.get('num_iters', 50)
        dry_run_iters = cfg.get('dry_run_iters', 10)
        block_counts = cfg.get('block_counts', None)  # None = auto
    else:
        sizes_mb = None  # Auto-generate sizes spanning L2 cache boundary
        patterns = ['seq_copy', 'seq_read', 'strided_copy', 'strided_read']
        dtypes = getattr(args, 'dtypes', None) or ['float32']
        num_iters = getattr(args, 'num_iters', 50)
        dry_run_iters = getattr(args, 'dry_run_iters', 10)
        block_counts = None  # Auto based on SM count

    bench = MemBwBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not getattr(args, 'use_cuda_events', False),
    )

    results = bench.run(
        sizes_mb=sizes_mb,
        patterns=patterns,
        dtypes=dtypes,
        block_counts=block_counts,
    )
    bench.print_summary(results)

    if getattr(args, 'output', None):
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(args.output, f'membw_{timestamp}.csv')
        bench.save_csv(results, csv_path)
        # Generate heatmap plot
        plot_path = os.path.join(args.output, f'membw_heatmap_{timestamp}.png')
        bench.plot_heatmap(results, plot_path)

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
        '--mode',
        choices=['all', 'gemm', 'membw'],
        default='all',
        help="Benchmark mode: 'all' (default), 'gemm', or 'membw'",
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        '--sizes',
        type=parse_size,
        nargs='+',
        default=None,
        metavar='M,N,K',
        help="GEMM sizes (e.g. --sizes 4096,4096,4096 64,4096,4096)",
    )
    parser.add_argument(
        '--dtypes',
        nargs='+',
        default=None,
        choices=['float32', 'float16', 'bfloat16', 'int8', 'float8_e4m3fn'],
        help="Data types to test",
    )
    parser.add_argument(
        '--num_iters',
        type=int,
        default=30,
        help="Number of measurement iterations (default: 30)",
    )
    parser.add_argument(
        '--dry_run_iters',
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        '--use_cuda_events',
        action='store_true',
        help="Force CUDA Events timing (skip CUPTI even if available)",
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Directory to save CSV results",
    )

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device available. Exiting.")
        sys.exit(1)

    # Print device info
    print_device_info()

    # Load config if provided
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            print(f"[INFO] Loaded config from: {args.config}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            sys.exit(1)

    # Run benchmarks
    if args.mode in ('all', 'gemm'):
        run_gemm(args, config)

    if args.mode in ('all', 'membw'):
        run_membw(args, config)

    print("\n[INFO] Benchmark complete.")


if __name__ == '__main__':
    main()
