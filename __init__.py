"""
xpu_benchmark: GPU performance benchmarking library.

Provides:
- GemmBenchmark: GEMM (matrix multiply) performance testing
- MemBwBenchmark: Memory bandwidth testing
- bench_gpu_time: Low-level GPU timing utility
"""

from .bench_gemm import (
    GemmBenchmark, GemmResult,
    LLMGemmBenchmark, LLMGemmResult,
    MODEL_SHAPE,
)
from .bench_memory import MemBwBenchmark, MemBwResult
from .bench_comm import CommBenchmark, CommBwResult
from .timing import bench_gpu_time
from .hw_spec import DEVICE_SPECS, get_peak_tflops, get_peak_bandwidth

__all__ = [
    "GemmBenchmark",
    "GemmResult",
    "LLMGemmBenchmark",
    "LLMGemmResult",
    "MODEL_SHAPE",
    "MemBwBenchmark",
    "MemBwResult",
    "CommBenchmark",
    "CommBwResult",
    "bench_gpu_time",
    "DEVICE_SPECS",
    "get_peak_tflops",
    "get_peak_bandwidth",
]
