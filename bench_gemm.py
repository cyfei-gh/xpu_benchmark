"""
GEMM (General Matrix Multiply) benchmark for GPU.

Measures:
- Execution time (ms)
- TFLOPS throughput
- MFU (Model Flops Utilization) relative to theoretical peak
- Memory bandwidth (GB/s)

Supports dtypes: float32, float16, bfloat16, int8, float8_e4m3fn
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .timing import bench_gpu_time
from .hw_spec import (
    DTYPE_FROM_STR, DTYPE_OUTPUT_MAPPING, DTYPE_BYTES,
    get_peak_tflops, get_peak_bandwidth,
)


@dataclass
class GemmResult:
    """Result of a single GEMM benchmark run."""
    m: int
    n: int
    k: int
    dtype: str
    median_time_ms: float
    std_time_ms: float
    tflops: float
    hw_tflops: float
    theory_tflops: float     # Theoretical peak TFLOPS for this dtype
    theory_time_ms: float    # Theoretical minimum time (ms) = max(compute_time, memory_time)
    mfu: float               # Model Flops Utilization (0~1)
    bandwidth_gbps: float
    hw_bandwidth: float
    mbu: float    # Bandwidth utilization (0~1)
    device_name: str

    def __str__(self) -> str:
        mfu_str = f"{self.mfu * 100:.1f}%" if self.mfu > 0 else "N/A"
        bw_str = f"{self.mbu * 100:.1f}%" if self.mbu > 0 else "N/A"
        theory_t_str = f"{self.theory_time_ms:.3f}" if self.theory_time_ms > 0 else "N/A"
        return (
            f"M={self.m:6d} N={self.n:6d} K={self.k:6d} | "
            f"dtype={self.dtype:<12s} | "
            f"time={self.median_time_ms:.3f}±{self.std_time_ms:.3f}ms | "
            f"theory_time={theory_t_str}ms | "
            f"TFLOPS={self.tflops:.2f} theory={self.theory_tflops:.2f} (MFU={mfu_str}) | "
            f"BW={self.bandwidth_gbps:.1f}GB/s (util={bw_str})"
        )


def _compute_gemm_flops(m: int, n: int, k: int) -> int:
    """Compute FLOPs for a GEMM: C = A @ B, A(m,k), B(k,n) -> C(m,n)."""
    return 2 * m * n * k


def _compute_gemm_bytes(
    m: int, n: int, k: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> int:
    """Compute total bytes transferred for a GEMM (read A, read B, write C)."""
    in_bytes = DTYPE_BYTES.get(input_dtype, 2)
    out_bytes = DTYPE_BYTES.get(output_dtype, 2)
    bytes_a = m * k * in_bytes
    bytes_b = k * n * in_bytes
    bytes_c = m * n * out_bytes
    return bytes_a + bytes_b + bytes_c


def _create_gemm_tensors(
    m: int, n: int, k: int,
    dtype: torch.dtype,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create input tensors for GEMM benchmark."""
    torch.manual_seed(42)

    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
    elif dtype == torch.int8:
        a = torch.randint(-128, 127, (m, k), device=device, dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), device=device, dtype=torch.int8)
    elif dtype == torch.float8_e4m3fn:
        # Create FP8 tensors via cast from float16
        a_f16 = torch.randn(m, k, device=device, dtype=torch.float16)
        b_f16 = torch.randn(k, n, device=device, dtype=torch.float16)
        a = a_f16.to(torch.float8_e4m3fn)
        b = b_f16.to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return a, b


def _run_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute GEMM for the given dtype."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.matmul(a, b)
    elif dtype == torch.int8:
        # Use torch._int_mm for int8 (returns int32)
        return torch._int_mm(a, b)
    elif dtype == torch.float8_e4m3fn:
        # Use torch._scaled_mm for FP8
        scale_a = torch.tensor(1.0, device=a.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=b.device, dtype=torch.float32)
        result, _ = torch._scaled_mm(
            a, b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        return result
    else:
        raise ValueError(f"Unsupported dtype for GEMM: {dtype}")


class GemmBenchmark:
    """
    GPU GEMM performance benchmark.

    Example usage:
        bench = GemmBenchmark()
        results = bench.run(
            sizes=[(4096, 4096, 4096), (64, 4096, 4096)],
            dtypes=['bfloat16', 'float16'],
        )
        bench.print_results(results)
        bench.save_csv(results, 'gemm_results.csv')
    """

    def __init__(
        self,
        device: str = 'cuda',
        num_iters: int = 30,
        dry_run_iters: int = 5,
        enable_cupti: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self.device = device
        self.num_iters = num_iters
        self.dry_run_iters = dry_run_iters
        self.enable_cupti = enable_cupti
        self.device_name = torch.cuda.get_device_name(0)

    def run_single(
        self,
        m: int,
        n: int,
        k: int,
        dtype_str: str = 'bfloat16',
    ) -> Optional[GemmResult]:
        """
        Run a single GEMM benchmark.

        Args:
            m, n, k: Matrix dimensions. C(m,n) = A(m,k) @ B(k,n)
            dtype_str: Data type string, e.g. 'bfloat16', 'float16', 'float32',
                       'int8', 'float8_e4m3fn'

        Returns:
            GemmResult or None on failure.
        """
        if dtype_str not in DTYPE_FROM_STR:
            print(f"[ERROR] Unsupported dtype: {dtype_str}")
            return None

        input_dtype = DTYPE_FROM_STR[dtype_str]
        output_dtype = DTYPE_OUTPUT_MAPPING["nvidia"].get(input_dtype, input_dtype)

        try:
            a, b = _create_gemm_tensors(m, n, k, input_dtype, self.device)

            def fn():
                return _run_gemm(a, b, input_dtype)

            real_ms, std_ms = bench_gpu_time(
                fn,
                enable_cupti=self.enable_cupti,
                num_iters=self.num_iters,
                dry_run_iters=self.dry_run_iters,
            )

        except Exception as e:
            print(f"[ERROR] GEMM failed M={m} N={n} K={k} dtype={dtype_str}: {e}")
            return None

        # Compute performance metrics
        flops = _compute_gemm_flops(m, n, k)
        tflops = (flops / 1e9) / real_ms
        total_bytes = _compute_gemm_bytes(m, n, k, input_dtype, output_dtype)
        bandwidth_gbps = (total_bytes / 1e6) / real_ms

        # MFU and bandwidth utilization
        hw_tflops = get_peak_tflops(self.device_name, dtype_str)
        hw_bandwidth = get_peak_bandwidth(self.device_name)
        theory_time_ms = max((flops / hw_tflops / 1e9), (total_bytes / hw_bandwidth / 1e6)) # ms
        theory_tflops = (flops / 1e9) / theory_time_ms

        mfu = (theory_time_ms / real_ms) if real_ms > 0 else 0.0
        bw_util = (bandwidth_gbps / hw_bandwidth) if hw_bandwidth > 0 else 0.0

        return GemmResult(
            m=m, n=n, k=k,
            dtype=dtype_str,
            median_time_ms=real_ms,
            std_time_ms=std_ms,
            tflops=tflops,
            hw_tflops=hw_tflops,
            theory_time_ms=theory_time_ms,
            theory_tflops=theory_tflops,
            mfu=mfu,
            bandwidth_gbps=bandwidth_gbps,
            hw_bandwidth=hw_bandwidth,
            mbu=bw_util,
            device_name=self.device_name,
        )

    def run(
        self,
        sizes: List[Tuple[int, int, int]],
        dtypes: List[str] = None,
    ) -> List[GemmResult]:
        """
        Run GEMM benchmarks for multiple sizes and dtypes.

        Args:
            sizes: List of (M, N, K) tuples.
            dtypes: List of dtype strings. Defaults to ['float32', 'bfloat16', 'float16'].

        Returns:
            List of GemmResult.
        """
        if dtypes is None:
            dtypes = ['float32', 'bfloat16', 'float16']

        results = []
        print(f"\n{'='*100}")
        print(f"GEMM Benchmark | Device: {self.device_name}")
        print(f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"{'='*100}")
        print(f"{'M':>6} {'N':>6} {'K':>6} | {'dtype':<12} | "
              f"{'time(ms)':>12} | {'theory_time(ms)':>9} | {'TFLOPS':>8} | {'theory_TFLOPS':>9} | {'MFU':>7} | "
              f"{'BW(GB/s)':>10} | {'BW_util':>8}")
        print(f"{'-'*120}")

        for dtype_str in dtypes:
            for m, n, k in sizes:
                result = self.run_single(m, n, k, dtype_str)
                if result is not None:
                    results.append(result)
                    mfu_str = f"{result.mfu*100:.1f}%" if result.mfu > 0 else "N/A"
                    bw_str = f"{result.mbu*100:.1f}%" if result.mbu > 0 else "N/A"
                    th_t_str = f"{result.theory_time_ms:.3f}" if result.theory_time_ms > 0 else "N/A"
                    print(
                        f"{result.m:>6} {result.n:>6} {result.k:>6} | "
                        f"{result.dtype:<12} | "
                        f"{result.median_time_ms:>8.3f}±{result.std_time_ms:.3f} | "
                        f"{th_t_str:>9} | "
                        f"{result.tflops:>8.2f} | "
                        f"{result.theory_tflops:>9.2f} | "
                        f"{mfu_str:>7} | "
                        f"{result.bandwidth_gbps:>10.1f} | "
                        f"{bw_str:>8}"
                    )
                else:
                    print(f"{m:>6} {n:>6} {k:>6} | {dtype_str:<12} | {'FAILED':>12}")

        print(f"{'='*100}")
        return results

    def print_summary(self, results: List[GemmResult]):
        """Print a summary of benchmark results."""
        if not results:
            print("No results to summarize.")
            return

        print(f"\n{'='*60}")
        print("GEMM Benchmark Summary")
        print(f"{'='*60}")

        # Group by dtype
        dtypes = sorted(set(r.dtype for r in results))
        for dtype in dtypes:
            dtype_results = [r for r in results if r.dtype == dtype]
            best = max(dtype_results, key=lambda r: r.tflops)
            print(f"\n[{dtype}]")
            print(f"  Best TFLOPS : {best.tflops:.2f} (M={best.m}, N={best.n}, K={best.k})")
            print(f"  Peak TFLOPS : {best.hw_tflops:.2f}")

    def save_csv(self, results: List[GemmResult], path: str):
        """Save benchmark results to CSV file."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'dtype', 'M', 'N', 'K',
                    'median_time_ms', 'std_time_ms', 'tflops',
                    'theory_time_ms', 'theory_tflops', 'mfu_pct',
                    'bandwidth_gbps', 'hw_bandwidth', 'mbu_pct',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.dtype, r.m, r.n, r.k,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}", f"{r.tflops:.4f}",
                        f"{r.theory_time_ms:.4f}", f"{r.theory_tflops:.4f}", f"{r.mfu*100:.2f}",
                        f"{r.bandwidth_gbps:.4f}", f"{r.hw_bandwidth:.4f}",
                        f"{r.mbu*100:.2f}",
                    ])
            print(f"[INFO] Results saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
