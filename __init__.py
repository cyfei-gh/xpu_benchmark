"""
xpu_benchmark: GPU performance benchmarking library.

Provides:
- GemmBenchmark: GEMM (matrix multiply) performance testing
- MemBwBenchmark: Memory bandwidth testing
- bench_gpu_time: Low-level GPU timing utility

Quick start:
    from xpu_benchmark import GemmBenchmark, MemBwBenchmark

    # GEMM benchmark
    gemm = GemmBenchmark()
    results = gemm.run(
        sizes=[(4096, 4096, 4096), (64, 4096, 4096)],
        dtypes=['bfloat16', 'float16'],
    )
    gemm.print_summary(results)
    gemm.save_csv(results, 'gemm_results.csv')

    # Memory bandwidth benchmark
    membw = MemBwBenchmark()
    results = membw.run(
        sizes_mb=[64, 256, 1024],
        patterns=['copy', 'triad'],
        dtypes=['float32', 'bfloat16'],
    )
    membw.print_summary(results)
    membw.save_csv(results, 'membw_results.csv')
"""

from .bench_gemm import GemmBenchmark, GemmResult
from .bench_membw import MemBwBenchmark, MemBwResult
from .timing import bench_gpu_time
from .hw_spec import DEVICE_SPECS, get_peak_tflops, get_peak_bandwidth

__all__ = [
    "GemmBenchmark",
    "GemmResult",
    "MemBwBenchmark",
    "MemBwResult",
    "bench_gpu_time",
    "DEVICE_SPECS",
    "get_peak_tflops",
    "get_peak_bandwidth",
]
