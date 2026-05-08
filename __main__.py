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
from typing import Optional

import torch
from datetime import datetime

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xpu_benchmark import GemmBenchmark, MemBwBenchmark, CommBenchmark, get_device_prefix
from xpu_benchmark import xpu_device as xpu


# ---------------------------------------------------------------------
# CUDA Driver API (cuda-python) 兼容导入：仅在 CUDA 可用时尝试
#   - cuda-python >= 12.8 : from cuda.bindings import driver
#   - cuda-python <  12.8 : from cuda import cuda as driver
# ---------------------------------------------------------------------
_cu = None  # type: ignore
_HAS_CUDA_PYTHON = False
if torch.cuda.is_available():
    try:
        from cuda.bindings import driver as _cu   # 新版 (>=12.8)
        _HAS_CUDA_PYTHON = True
    except ImportError:
        try:
            from cuda import cuda as _cu          # 老版
            _HAS_CUDA_PYTHON = True
        except ImportError:
            _HAS_CUDA_PYTHON = False


def _cu_check(res_tuple):
    """检查 CUDA Driver API 返回值；成功则返回除 CUresult 外的值。"""
    if not isinstance(res_tuple, tuple):
        res, rest = res_tuple, ()
    else:
        res, *rest = res_tuple

    if int(res) != int(_cu.CUresult.CUDA_SUCCESS):
        err_str = _cu.cuGetErrorString(res)
        if isinstance(err_str, tuple):
            err_str = err_str[1]
        if isinstance(err_str, bytes):
            err_str = err_str.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA Driver API error: {err_str} (code={int(res)})")

    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


def _cu_get_attr(dev, attr) -> int:
    """查询设备属性；失败返回 -1。"""
    try:
        return int(_cu_check(_cu.cuDeviceGetAttribute(attr, dev)))
    except Exception:
        return -1


def _cu_arch_name(major: int, minor: int) -> str:
    """根据 compute capability 返回架构名称。"""
    if major == 7:
        return {
            0: "Volta (V100)",
            2: "Volta (Xavier)",
            5: "Turing (T4/RTX20)",
        }.get(minor, "Volta/Turing")
    if major == 8:
        return {
            0: "Ampere (A100)",
            6: "Ampere (A10/A30/A40/RTX30)",
            7: "Ampere (Orin)",
            9: "Ada Lovelace (L20/L40/RTX40)",
        }.get(minor, "Ampere/Ada")
    if major == 9:
        return "Hopper (H100/H20/H200)" if minor == 0 else "Hopper"
    if major == 10:
        return {
            0: "Blackwell Datacenter (B200/B100)",
            1: "Blackwell (GB10)",
        }.get(minor, "Blackwell")
    if major == 12:
        return "Blackwell (RTX50)"
    return "Unknown"


def _tensor_memory_per_sm_bytes(major: int, minor: int) -> int:
    """Blackwell (SM 10.0+) 每 SM 固定 256 KB TMEM；早期架构无此硬件。"""
    if major >= 10:
        return 256 * 1024
    return 0


def _unified_l1_smem_per_sm_bytes(major: int, minor: int) -> int:
    """每 SM 上 L1 与 Shared Memory 合并的 Unified On-chip SRAM 总大小（硬件规格）。"""
    if major == 7 and minor == 0:
        return 128 * 1024            # V100
    if major == 7 and minor == 5:
        return 96 * 1024             # Turing
    if major == 8 and minor == 0:
        return 192 * 1024            # A100
    if major == 8 and minor in (6, 7):
        return 128 * 1024            # GA10x / Orin
    if major == 8 and minor == 9:
        return 128 * 1024            # Ada (L20/L40/RTX40)
    if major == 9 and minor == 0:
        return 256 * 1024            # Hopper (H100/H20)
    if major >= 10:
        return 256 * 1024            # Blackwell (B200 等, 估计值)
    return 0


def _print_size(label: str, nbytes: Optional[int]) -> None:
    if nbytes is None or nbytes <= 0:
        print(f"  {label:<45s} : N/A")
        return
    if nbytes >= (1 << 20):
        print(f"  {label:<45s} : {nbytes:>10d} B  ({nbytes / 1024 / 1024:.2f} MB)")
    elif nbytes >= (1 << 10):
        print(f"  {label:<45s} : {nbytes:>10d} B  ({nbytes / 1024:.2f} KB)")
    else:
        print(f"  {label:<45s} : {nbytes:>10d} B")


def _print_cuda_device0_spec() -> None:
    """通过 CUDA Driver API 打印 device 0 的详细规格（L2/SMEM/L1/TMEM/其它）。"""
    if not _HAS_CUDA_PYTHON:
        print("[HINT] 未安装 cuda-python，跳过 L2/SMEM/L1 等详细规格查询。")
        print("[HINT] 可执行: pip install cuda-python")
        return

    _cu_check(_cu.cuInit(0))
    dev = _cu_check(_cu.cuDeviceGet(0))

    A = _cu.CUdevice_attribute
    cc_major = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    cc_minor = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    sm_count = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    total_mem = int(_cu_check(_cu.cuDeviceTotalMem(dev)))
    driver_version = int(_cu_check(_cu.cuDriverGetVersion()))

    print(f"  CUDA Driver Ver.: {driver_version // 1000}.{(driver_version % 1000) // 10}")
    print(f"  Architecture    : {cc_major}.{cc_minor} ({_cu_arch_name(cc_major, cc_minor)})")
    print(f"  Global Memory   : {total_mem / (1024 ** 3):.2f} GB")
    print(f"  SM Count        : {sm_count}")
    print("-" * 60)

    # ============ L2 Cache ============
    l2_bytes       = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    l2_max_persist = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)
    l2_max_access  = -1
    if hasattr(A, "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE"):
        l2_max_access = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)

    print("[L2 Cache]")
    _print_size("L2 Cache Size (device-wide)",      l2_bytes)
    _print_size("L2 Max Persisting Cache Size",     l2_max_persist)
    _print_size("L2 Max Access Policy Window Size", l2_max_access)

    # ============ Shared Memory ============
    smem_per_block        = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    smem_per_block_optin  = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    smem_per_sm           = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    smem_reserved_per_blk = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)

    print("\n[Shared Memory]")
    _print_size("Max Shared Memory per Block (default)", smem_per_block)
    _print_size("Max Shared Memory per Block (opt-in)",  smem_per_block_optin)
    _print_size("Max Shared Memory per SM",              smem_per_sm)
    _print_size("Reserved Shared Memory per Block",      smem_reserved_per_blk)
    total_smem = smem_per_sm * sm_count if smem_per_sm > 0 and sm_count > 0 else 0
    _print_size("Total Shared Memory (per SM * SM count)", total_smem)

    # ============ L1 Cache / Unified L1+SMEM ============
    print("\n[L1 Cache / Unified L1+SharedMem]")
    unified_bytes = _unified_l1_smem_per_sm_bytes(cc_major, cc_minor)
    _print_size("Unified L1+SMEM per SM (HW total, from arch)", unified_bytes)
    l1_approx = max(0, unified_bytes - max(0, smem_per_sm))
    _print_size("L1 Cache per SM (approx = Unified - SMEM)", l1_approx)
    total_l1 = l1_approx * sm_count if sm_count > 0 else 0
    _print_size("Total L1 Cache (approx, across all SMs)", total_l1)

    global_l1 = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)
    local_l1  = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)
    print(f"  {'Global Loads L1 Cached':<45s} : {'Yes' if global_l1 > 0 else 'No'}")
    print(f"  {'Local Loads L1 Cached':<45s} : {'Yes' if local_l1  > 0 else 'No'}")

    # ============ Tensor Memory (Blackwell+) ============
    print("\n[Tensor Memory (TMEM, Blackwell+)]")
    tmem_per_sm = _tensor_memory_per_sm_bytes(cc_major, cc_minor)
    if tmem_per_sm == 0:
        print(f"  {'Tensor Memory per SM':<45s} : "
              f"N/A (not supported on this architecture)")
    else:
        _print_size("Tensor Memory per SM", tmem_per_sm)
        _print_size("Total Tensor Memory (per SM * SM count)",
                    tmem_per_sm * sm_count)

    # ============ 其它信息 ============
    regs_per_block     = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    regs_per_sm        = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    warp_size          = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    max_threads_per_sm = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    mem_bus_width      = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    mem_clock_khz      = _cu_get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    print("\n[Other Info]")
    print(f"  {'Warp Size':<45s} : {warp_size}")
    print(f"  {'Max Threads per SM':<45s} : {max_threads_per_sm}")
    print(f"  {'Max Registers per Block':<45s} : {regs_per_block} (4-byte regs)")
    print(f"  {'Max Registers per SM':<45s} : {regs_per_sm} (4-byte regs)")
    print(f"  {'Global Memory Bus Width':<45s} : {mem_bus_width} bits")
    if mem_clock_khz > 0:
        eff_ghz = mem_clock_khz / 1e6 * 2.0  # DDR -> x2
        bw_gbs  = mem_bus_width / 8.0 * mem_clock_khz * 1e3 * 2.0 / 1e9
        print(f"  {'Memory Clock Rate':<45s} : {eff_ghz:.2f} GHz (effective)")
        print(f"  {'Theoretical Peak HBM/GDDR Bandwidth':<45s} : {bw_gbs:.2f} GB/s")

def _print_npu_device0_spec(dev_id: Optional[int] = None) -> bool:
    """
    尝试通过 torch_npu 打印 NPU 设备的简要规格。
    成功返回 True, 否则返回 False。
    """
    try:
        import torch
        import torch_npu  # noqa: F401  # 导入后 torch.npu 才挂载
    except Exception as e:
        sys.stderr.write(f"[INFO] torch_npu 不可用: {e}\n")
        return False

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        sys.stderr.write("[INFO] 未检测到可用的 NPU 设备。\n")
        return False

    print(f"torch_npu Version : {torch_npu.__version__}")

    return True


def print_device_info():
    """Print GPU/NPU device 0 information (only device 0)."""
    if not xpu.is_available():
        print("[ERROR] No XPU (CUDA / NPU) device available.")
        return

    backend = xpu.backend()
    name = xpu.get_device_name(0)
    props = xpu.get_device_properties(0)
    print(f"\n{'='*60}")
    print(f"  Device Name     : {name}")
    print(f"  Device Count    : {xpu.device_count()}")
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  Props           : {props}")
    print("-" * 60)

    # 追加：通过 CUDA Driver API 打印 device 0 的 L2/SMEM/L1/TMEM/其它详细规格
    if backend == 'cuda':
        _print_cuda_device0_spec()
    elif backend == 'npu':
        _print_npu_device0_spec()
    else:
        print("[INFO] 未实现的 XPU 后端:", backend)

    print(f"{'='*60}\n")


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_llm_gemm(config: dict, output_dir: str = None, use_events: bool = False):
    """Run LLM GEMM benchmark (QKV, Proj, FFN, MoE workloads) based on config."""
    cfg = config['llm_gemm']
    model_name = cfg.get('model', 'HY-image-3.0')
    batch_sizes = cfg.get('batch_sizes', [1])
    dtypes = cfg.get('dtypes', ['bfloat16'])
    tp = cfg.get('tp', 1)
    num_iters = cfg.get('num_iters', 30)
    dry_run_iters = cfg.get('dry_run_iters', 5)

    bench = GemmBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_events,
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
        device_prefix = get_device_prefix(bench.device_name)
        file_name = os.path.join(output_dir, f'{device_prefix}_gemm_{model_name}_{timestamp}')
        bench.save_csv(results, f'{file_name}.csv')
        bench.plot_batch_tflops_curve(results, f'{file_name}.png')

    return results


def run_membw(config: dict, output_dir: str = None, use_events: bool = False):
    """Run memory bandwidth benchmark based on config."""
    cfg = config['memory']
    sizes_mb = cfg.get('sizes_mb', None)
    patterns = cfg.get('patterns', ['seq_copy', 'strided_copy'])
    dtypes = cfg.get('dtypes', ['float32'])
    num_iters = cfg.get('num_iters', 50)
    dry_run_iters = cfg.get('dry_run_iters', 10)
    flush_l2_cache = cfg.get('flush_l2_cache', False)

    bench = MemBwBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
        enable_cupti=not use_events,
        flush_l2_cache=flush_l2_cache,
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
        device_prefix = get_device_prefix(bench.device_name)
        # 文件名按缓存策略区分：HBM（flush L2） / L2（保留 L2）
        cache_tag = 'HBM' if flush_l2_cache else 'L2'
        file_name = os.path.join(output_dir, f'{device_prefix}_membw_{cache_tag}_{timestamp}')
        bench.save_csv(results, f'{file_name}.csv')
        bench.plot_size_bw_curve(results, f'{file_name}.png')

    return results


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
        '--use_events',
        action='store_true',
        help="Force CUDA Events timing (skip CUPTI even if available)",
    )

    args = parser.parse_args()

    if not xpu.is_available():
        print("[ERROR] No XPU device available. Exiting.")
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
        run_llm_gemm(config, output_dir=args.output, use_events=args.use_events)

    if 'memory' in config:
        run_membw(config, output_dir=args.output, use_events=args.use_events)

    # if 'comm' in config:
    #     run_comm(config, output_dir=args.output)

    print("\n[INFO] Benchmark complete.")


if __name__ == '__main__':
    main()
