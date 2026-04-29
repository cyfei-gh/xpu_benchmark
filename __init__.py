"""
xpu_benchmark: GPU/NPU performance benchmarking library.

Provides:
- GemmBenchmark: GEMM (matrix multiply) performance testing
- MemBwBenchmark: Memory bandwidth testing
- CommBenchmark: Collective communication bandwidth testing
- bench_gpu_time: Low-level device timing utility

支持后端：NVIDIA CUDA 与 Ascend NPU (torch_npu)。
"""

from . import xpu_device
from .bench_gemm import (
    GemmBenchmark, GemmResult,
    MODEL_SHAPE,
)
from .bench_memory import MemBwBenchmark, MemBwResult
from .bench_comm import CommBenchmark, CommBwResult
from .timing import bench_gpu_time
from .hw_spec import DEVICE_SPECS, get_peak_tflops, get_peak_bandwidth

__all__ = [
    "xpu_device",
    "GemmBenchmark",
    "GemmResult",
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
