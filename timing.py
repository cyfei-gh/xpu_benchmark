"""
GPU timing utilities.

Supports two timing methods:
1. CUPTI (Preferred): Hardware-level profiling, most accurate GPU kernel time.
   Requires cupti-python >= 13.0.0 (CUDA 13+).
2. CUDA Events (Fallback): Standard CUDA event timing, good accuracy.
   Automatically used if CUPTI is not available.
"""

import torch
import numpy as np
from typing import Callable, Tuple, Any, Optional

# Try to import CUPTI
try:
    import cupti
    if hasattr(cupti, 'ProfilerContext'):
        CUPTI_AVAILABLE = True
    else:
        CUPTI_AVAILABLE = False
except ImportError:
    CUPTI_AVAILABLE = False

_cupti_warning_shown = False


def _bench_with_cuda_events(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    num_iters: int = 30,
    dry_run_iters: int = 5,
    cold_l2_cache: bool = False,
) -> Tuple[float, float]:
    """
    Benchmark a GPU function using CUDA Events.

    Returns:
        (median_time_ms, std_time_ms)
    """
    if kwargs is None:
        kwargs = {}

    # Warmup
    for _ in range(dry_run_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times_ms = []

    if cold_l2_cache:
        # Rotate buffers to flush L2 cache between iterations
        # Allocate a large buffer to flush L2 (~40MB for L20)
        flush_size = 40 * 1024 * 1024 // 4  # 40MB in float32 elements
        flush_buf = torch.empty(flush_size, dtype=torch.float32, device='cuda')

    for _ in range(num_iters):
        if cold_l2_cache:
            # Touch the flush buffer to evict L2 cache
            flush_buf.fill_(0.0)
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        fn(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    times_arr = np.array(times_ms)
    return float(np.median(times_arr)), float(np.std(times_arr))


def _bench_with_cupti(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    num_iters: int = 30,
    dry_run_iters: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark a GPU function using CUPTI for hardware-level timing.

    Returns:
        (median_time_ms, std_time_ms)
    """
    if kwargs is None:
        kwargs = {}

    # Warmup
    for _ in range(dry_run_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_iters):
        with cupti.ProfilerContext() as ctx:
            fn(*args, **kwargs)
            torch.cuda.synchronize()
        kernel_times = ctx.get_kernel_times_ms()
        if kernel_times:
            times_ms.append(sum(kernel_times))

    if not times_ms:
        # Fallback if CUPTI returned no data
        return _bench_with_cuda_events(fn, args, kwargs, num_iters, dry_run_iters)

    times_arr = np.array(times_ms)
    return float(np.median(times_arr)), float(np.std(times_arr))


def bench_gpu_time(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    enable_cupti: bool = True,
    num_iters: int = 30,
    dry_run_iters: int = 5,
    cold_l2_cache: bool = False,
) -> Tuple[float, float]:
    """
    Benchmark a GPU function and return timing statistics.

    Automatically uses CUPTI if available and enabled, otherwise falls back
    to CUDA Events.

    Args:
        fn: The function to benchmark.
        args: Positional arguments to pass to fn.
        kwargs: Keyword arguments to pass to fn.
        enable_cupti: Whether to try CUPTI timing (default True).
                      Set to False to force CUDA Events.
        num_iters: Number of measurement iterations.
        dry_run_iters: Number of warmup iterations.
        cold_l2_cache: If True, flush L2 cache between iterations (CUDA Events only).

    Returns:
        (median_time_ms, std_time_ms): Median and std of kernel execution time in ms.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    global _cupti_warning_shown
    use_cupti = enable_cupti and CUPTI_AVAILABLE

    if enable_cupti and not CUPTI_AVAILABLE and not _cupti_warning_shown:
        _cupti_warning_shown = True
        try:
            import cupti as _cupti_check
            # Module exists but lacks ProfilerContext
            print("[WARNING] CUPTI module found but missing ProfilerContext API. "
                  "Try 'pip install -U cupti-python>=13.0.0' (requires CUDA 13+). "
                  "Falling back to CUDA events.")
        except ImportError:
            print("[WARNING] CUPTI is not installed. Try 'pip install -U cupti-python'. "
                  "Falling back to CUDA events.")

    if use_cupti:
        try:
            return _bench_with_cupti(fn, args, kwargs, num_iters, dry_run_iters)
        except Exception as e:
            print(f"[WARNING] CUPTI timing failed ({e}), falling back to CUDA events.")

    return _bench_with_cuda_events(fn, args, kwargs, num_iters, dry_run_iters, cold_l2_cache)
