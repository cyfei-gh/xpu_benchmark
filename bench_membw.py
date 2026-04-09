"""
GPU Memory Bandwidth benchmark using Triton kernels.

Measures effective memory bandwidth with fine-grained control over:
- Sequential (contiguous) memory access
- Strided (non-contiguous / scattered) memory access
- Number of thread blocks launched (via grid configuration)
- Data sizes spanning L2 cache boundary

Key metrics:
- Effective bandwidth (GB/s)
- Bandwidth utilization vs theoretical peak

Generates heatmap plots: block_count x data_size -> bandwidth,
highlighting L2 cache effects.
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .timing import bench_gpu_time
from .hw_spec import get_peak_bandwidth


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class MemBwResult:
    """Result of a single memory bandwidth benchmark run."""
    pattern: str          # e.g. 'seq_copy', 'seq_read', 'seq_write', 'strided_copy', 'strided_read'
    size_mb: float        # Buffer size in MB
    dtype: str
    num_blocks: int       # Number of thread blocks launched
    block_size: int       # Elements per block
    median_time_ms: float
    std_time_ms: float
    bandwidth_gbps: float
    peak_bandwidth_gbps: float
    utilization: float    # bandwidth / peak (0~1)
    in_l2_cache: bool     # Whether data fits in L2 cache
    device_name: str

    def __str__(self) -> str:
        util_str = f"{self.utilization*100:.1f}%" if self.utilization > 0 else "N/A"
        l2_str = "L2" if self.in_l2_cache else "HBM"
        return (
            f"pattern={self.pattern:<14} size={self.size_mb:8.2f}MB "
            f"blocks={self.num_blocks:>5} | "
            f"dtype={self.dtype:<8} | "
            f"time={self.median_time_ms:.3f}±{self.std_time_ms:.3f}ms | "
            f"BW={self.bandwidth_gbps:.1f}GB/s (util={util_str}) [{l2_str}]"
        )


# ===================================================================
# L2 cache size lookup
# ===================================================================

# L2 cache sizes in bytes for known GPU architectures
L2_CACHE_SIZES = {
    "L20": 96 * 1024 * 1024,        # 96 MB
    "H20": 60 * 1024 * 1024,        # 60 MB
    "H100": 50 * 1024 * 1024,       # 50 MB
    "A100": 40 * 1024 * 1024,       # 40 MB
    "RTX 5000": 96 * 1024 * 1024,   # 96 MB
    "RTX 4090": 72 * 1024 * 1024,   # 72 MB
}


def _get_l2_cache_size(device_name: str) -> int:
    """Get L2 cache size in bytes for the given device. Default 40MB if unknown."""
    for key, size in L2_CACHE_SIZES.items():
        if key.lower() in device_name.lower():
            return size
    return 40 * 1024 * 1024  # Conservative default


# ===================================================================
# Triton kernels for memory access patterns
# ===================================================================

@triton.jit
def _seq_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential copy: contiguous read from src + contiguous write to dst.
    Each program instance handles a chunk of BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def _seq_read_kernel(
    src_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential read: contiguous read, accumulate to prevent optimization.
    Each program instance reads BLOCK_SIZE elements and atomically adds the sum.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    # Reduce to scalar and atomic add to prevent dead code elimination
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(out_ptr, block_sum)


@triton.jit
def _seq_write_kernel(
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential write: contiguous write of constant value."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(dst_ptr + offsets, 1.0, mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr, dst_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided copy: gather read via indices + scatter write via indices.
    Non-contiguous memory access pattern to stress memory subsystem.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load scattered indices
    idx = tl.load(indices_ptr + offsets, mask=mask)
    # Gather read
    data = tl.load(src_ptr + idx, mask=mask)
    # Scatter write
    tl.store(dst_ptr + idx, data, mask=mask)


@triton.jit
def _strided_read_kernel(
    src_ptr, out_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided read: gather read via indices, reduce to scalar.
    Non-contiguous memory access pattern.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load scattered indices
    idx = tl.load(indices_ptr + offsets, mask=mask)
    # Gather read
    data = tl.load(src_ptr + idx, mask=mask)
    # Reduce to scalar and atomic add to prevent dead code elimination
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(out_ptr, block_sum)


# ===================================================================
# Benchmark class
# ===================================================================

class MemBwBenchmark:
    """
    GPU Memory Bandwidth benchmark using Triton kernels.

    Tests sequential (contiguous) and strided (non-contiguous) memory access
    patterns with varying data sizes and thread block counts. Generates heatmap
    plots showing bandwidth as a function of block count and data size,
    highlighting L2 cache boundary effects.

    Example usage:
        bench = MemBwBenchmark()
        results = bench.run(
            sizes_mb=[1, 4, 16, 64, 256, 1024],
            patterns=['seq_copy', 'strided_copy'],
            dtypes=['float32'],
        )
        bench.plot_heatmap(results, 'membw_heatmap.png')
        bench.save_csv(results, 'membw_results.csv')
    """

    def __init__(
        self,
        device: str = 'cuda',
        num_iters: int = 50,
        dry_run_iters: int = 10,
        enable_cupti: bool = True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self.device = device
        self.num_iters = num_iters
        self.dry_run_iters = dry_run_iters
        self.enable_cupti = enable_cupti
        self.device_name = torch.cuda.get_device_name(0)
        self.l2_cache_bytes = _get_l2_cache_size(self.device_name)
        self.l2_cache_mb = self.l2_cache_bytes / (1024 * 1024)

    def _dtype_from_str(self, dtype_str: str) -> torch.dtype:
        mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported dtype for memory benchmark: {dtype_str}. "
                             f"Supported: {list(mapping.keys())}")
        return mapping[dtype_str]

    def _bytes_per_element(self, dtype: torch.dtype) -> int:
        return {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(dtype, 4)

    def _get_default_block_counts(self) -> List[int]:
        """
        Generate a range of block counts to sweep.
        Uses SM count as reference for meaningful parallelism levels.
        """
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
        # Sweep from 1 SM to 8x SM count
        counts = sorted(set([
            1,
            max(1, sm_count // 4),
            max(1, sm_count // 2),
            sm_count,
            sm_count * 2,
            sm_count * 4,
            sm_count * 8,
        ]))
        return [c for c in counts if c >= 1]

    def _get_default_sizes_mb(self) -> List[float]:
        """
        Generate data sizes that span below, around, and above L2 cache.
        This reveals the L2 cache cliff effect.
        """
        l2_mb = self.l2_cache_mb
        sizes = sorted(set([
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            l2_mb * 0.1,
            l2_mb * 0.25,
            l2_mb * 0.5,
            l2_mb * 0.75,
            l2_mb * 1.0,
            l2_mb * 1.5,
            l2_mb * 2.0,
            l2_mb * 4.0,
            256.0,
            512.0,
            1024.0,
        ]))
        # Round to 2 decimal places and deduplicate
        sizes = sorted(set(round(s, 2) for s in sizes if s >= 0.25))
        return sizes

    def _generate_strided_indices(
        self, n_elements: int, stride: int,
    ) -> torch.Tensor:
        """
        Generate strided (non-contiguous) access indices.
        Uses a large stride to scatter accesses across memory,
        simulating non-coalesced access patterns.
        """
        # Create indices: 0, stride, 2*stride, ... mod n_elements
        # Use a prime-like stride to maximize cache line conflicts
        indices = torch.arange(n_elements, device=self.device, dtype=torch.int32)
        indices = (indices * stride) % n_elements
        return indices

    def _compute_block_size(self, n_elements: int, num_blocks: int) -> int:
        """
        Compute BLOCK_SIZE (elements per thread block) given total elements
        and desired number of blocks. BLOCK_SIZE must be a power of 2 for Triton.
        """
        raw = max(1, n_elements // num_blocks)
        # Round up to next power of 2 (Triton requires power-of-2 constexpr)
        power = 1
        while power < raw:
            power *= 2
        # Clamp to reasonable range [256, 65536]
        power = max(256, min(power, 65536))
        return power

    def run_single(
        self,
        size_mb: float,
        pattern: str = 'seq_copy',
        dtype_str: str = 'float32',
        num_blocks: int = None,
    ) -> Optional[MemBwResult]:
        """
        Run a single memory bandwidth benchmark with a Triton kernel.

        Args:
            size_mb: Buffer size in MB (per buffer).
            pattern: Access pattern. One of:
                     'seq_copy'     - Sequential copy: dst = src
                     'seq_read'     - Sequential read: sum(src)
                     'seq_write'    - Sequential write: dst = val
                     'strided_copy' - Strided copy (stride=128 elements)
                     'strided_read' - Strided read (stride=128 elements)
            dtype_str: Data type string.
            num_blocks: Number of thread blocks to launch.
                        If None, auto-compute based on data size.

        Returns:
            MemBwResult or None on failure.
        """
        try:
            dtype = self._dtype_from_str(dtype_str)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return None

        bpe = self._bytes_per_element(dtype)
        n_elements = int(size_mb * 1024 * 1024 / bpe)
        # Ensure n_elements is reasonable
        n_elements = max(1024, n_elements)

        if num_blocks is None:
            num_blocks = max(1, n_elements // 1024)

        # Compute BLOCK_SIZE (power of 2, elements per block)
        block_size = self._compute_block_size(n_elements, num_blocks)

        # Recompute actual grid size: ceil(n_elements / block_size)
        grid_size = (n_elements + block_size - 1) // block_size

        # Stride for non-contiguous patterns (large prime for scattering)
        stride = 128

        try:
            # Allocate buffers
            src = torch.randn(n_elements, dtype=dtype, device=self.device)
            dst = torch.empty(n_elements, dtype=dtype, device=self.device)
            out_scalar = torch.zeros(1, dtype=dtype, device=self.device)

            # Pre-generate strided indices if needed
            indices = None
            if pattern in ('strided_copy', 'strided_read'):
                indices = self._generate_strided_indices(n_elements, stride)

            # Build the benchmark function with Triton kernel launch
            if pattern == 'seq_copy':
                def fn():
                    _seq_copy_kernel[(grid_size,)](
                        src, dst, n_elements, BLOCK_SIZE=block_size,
                    )
                bytes_transferred = 2 * n_elements * bpe  # read + write
            elif pattern == 'seq_read':
                def fn():
                    out_scalar.zero_()
                    _seq_read_kernel[(grid_size,)](
                        src, out_scalar, n_elements, BLOCK_SIZE=block_size,
                    )
                bytes_transferred = n_elements * bpe  # read only
            elif pattern == 'seq_write':
                def fn():
                    _seq_write_kernel[(grid_size,)](
                        dst, n_elements, BLOCK_SIZE=block_size,
                    )
                bytes_transferred = n_elements * bpe  # write only
            elif pattern == 'strided_copy':
                def fn():
                    _strided_copy_kernel[(grid_size,)](
                        src, dst, indices, n_elements, BLOCK_SIZE=block_size,
                    )
                bytes_transferred = 2 * n_elements * bpe  # read + write
            elif pattern == 'strided_read':
                def fn():
                    out_scalar.zero_()
                    _strided_read_kernel[(grid_size,)](
                        src, out_scalar, indices, n_elements, BLOCK_SIZE=block_size,
                    )
                bytes_transferred = n_elements * bpe  # read only
            else:
                print(f"[ERROR] Unknown pattern: {pattern}. "
                      f"Supported: seq_copy, seq_read, seq_write, strided_copy, strided_read")
                return None

            # Determine if data fits in L2 cache
            in_l2 = True # (n_elements * bpe) <= self.l2_cache_bytes

            # Flush L2 cache for HBM-bound tests to get realistic bandwidth
            median_ms, std_ms = bench_gpu_time(
                fn,
                enable_cupti=self.enable_cupti,
                num_iters=self.num_iters,
                dry_run_iters=self.dry_run_iters,
                cold_l2_cache=(not in_l2),  # Flush L2 for HBM-bound tests
            )

        except Exception as e:
            print(f"[ERROR] MemBw failed pattern={pattern} size={size_mb}MB "
                  f"blocks={num_blocks} dtype={dtype_str}: {e}")
            return None

        # Compute bandwidth
        time_s = median_ms / 1000.0
        bandwidth_gbps = (bytes_transferred / 1e9) / time_s if time_s > 0 else 0.0

        peak_bw = get_peak_bandwidth(self.device_name)
        utilization = (bandwidth_gbps / peak_bw) if peak_bw > 0 else 0.0

        return MemBwResult(
            pattern=pattern,
            size_mb=round(n_elements * bpe / (1024 * 1024), 2),
            dtype=dtype_str,
            num_blocks=grid_size,
            block_size=block_size,
            median_time_ms=median_ms,
            std_time_ms=std_ms,
            bandwidth_gbps=bandwidth_gbps,
            peak_bandwidth_gbps=peak_bw,
            utilization=utilization,
            in_l2_cache=in_l2,
            device_name=self.device_name,
        )

    def run(
        self,
        sizes_mb: List[float] = None,
        patterns: List[str] = None,
        dtypes: List[str] = None,
        block_counts: List[int] = None,
    ) -> List[MemBwResult]:
        """
        Run memory bandwidth benchmarks sweeping block counts and data sizes.

        Args:
            sizes_mb: Buffer sizes in MB. Defaults to auto-generated range
                      spanning L2 cache boundary.
            patterns: Access patterns. Defaults to ['seq_copy', 'strided_copy'].
            dtypes: Data types. Defaults to ['float32'].
            block_counts: Number of blocks to sweep. Defaults to auto range
                          based on SM count.

        Returns:
            List of MemBwResult.
        """
        if sizes_mb is None:
            sizes_mb = self._get_default_sizes_mb()
        if patterns is None:
            patterns = ['seq_copy', 'strided_copy']
        if dtypes is None:
            dtypes = ['float32']
        if block_counts is None:
            block_counts = self._get_default_block_counts()

        results = []
        print(f"\n{'='*120}")
        print(f"Memory Bandwidth Benchmark (Triton Kernels) | Device: {self.device_name}")
        print(f"L2 Cache: {self.l2_cache_mb:.0f} MB | "
              f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"Block counts: {block_counts}")
        print(f"Data sizes (MB): {[f'{s:.2f}' for s in sizes_mb]}")
        print(f"{'='*120}")
        print(f"{'pattern':<14} {'size(MB)':>8} {'blocks':>7} | {'dtype':<8} | "
              f"{'time(ms)':>12} | {'BW(GB/s)':>10} | {'util':>7} | {'L2?':>4}")
        print(f"{'-'*120}")

        for dtype_str in dtypes:
            for pattern in patterns:
                for size_mb in sizes_mb:
                    for nb in block_counts:
                        result = self.run_single(size_mb, pattern, dtype_str, nb)
                        if result is not None:
                            results.append(result)
                            util_str = (f"{result.utilization*100:.1f}%"
                                        if result.utilization > 0 else "N/A")
                            l2_str = "Y" if result.in_l2_cache else "N"
                            print(
                                f"{result.pattern:<14} "
                                f"{result.size_mb:>8.2f} "
                                f"{result.num_blocks:>7} | "
                                f"{result.dtype:<8} | "
                                f"{result.median_time_ms:>8.3f}±"
                                f"{result.std_time_ms:.3f} | "
                                f"{result.bandwidth_gbps:>10.1f} | "
                                f"{util_str:>7} | "
                                f"{l2_str:>4}"
                            )
                        else:
                            print(f"{pattern:<14} {size_mb:>8.2f} "
                                  f"{nb:>7} | {dtype_str:<8} | {'FAILED':>12}")

        print(f"{'='*120}")
        return results

    def print_summary(self, results: List[MemBwResult]):
        """Print a summary of memory bandwidth results."""
        if not results:
            print("No results to summarize.")
            return

        print(f"\n{'='*70}")
        print(f"Memory Bandwidth Summary | L2 Cache: {self.l2_cache_mb:.0f} MB")
        print(f"{'='*70}")

        patterns = sorted(set(r.pattern for r in results))
        for pattern in patterns:
            pat_results = [r for r in results if r.pattern == pattern]

            # Split by L2 vs HBM
            l2_results = [r for r in pat_results if r.in_l2_cache]
            hbm_results = [r for r in pat_results if not r.in_l2_cache]

            print(f"\n[{pattern}]")
            if l2_results:
                best_l2 = max(l2_results, key=lambda r: r.bandwidth_gbps)
                print(f"  Best BW (L2-resident) : {best_l2.bandwidth_gbps:.1f} GB/s "
                      f"(size={best_l2.size_mb:.2f}MB, blocks={best_l2.num_blocks})")
            if hbm_results:
                best_hbm = max(hbm_results, key=lambda r: r.bandwidth_gbps)
                print(f"  Best BW (HBM)         : {best_hbm.bandwidth_gbps:.1f} GB/s "
                      f"(size={best_hbm.size_mb:.2f}MB, blocks={best_hbm.num_blocks})")
            if pat_results:
                best = max(pat_results, key=lambda r: r.bandwidth_gbps)
                print(f"  Peak BW (device)      : {best.peak_bandwidth_gbps:.1f} GB/s")
                if best.utilization > 0:
                    print(f"  Best utilization      : {best.utilization*100:.1f}%")

    def plot_heatmap(
        self,
        results: List[MemBwResult],
        output_path: str = 'membw_heatmap.png',
        pattern_filter: str = None,
    ):
        """
        Generate heatmap plots: block_count x data_size -> bandwidth.

        Creates one subplot per pattern, with L2 cache boundary marked.

        Args:
            results: List of MemBwResult from run().
            output_path: Path to save the figure.
            pattern_filter: If set, only plot this pattern.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping heatmap plot. "
                  "Install with: pip install matplotlib")
            return

        patterns = sorted(set(r.pattern for r in results))
        if pattern_filter:
            patterns = [p for p in patterns if p == pattern_filter]

        if not patterns:
            print("[WARNING] No matching patterns found for plotting.")
            return

        n_patterns = len(patterns)
        fig, axes = plt.subplots(1, n_patterns, figsize=(8 * n_patterns, 6),
                                 squeeze=False)

        for idx, pattern in enumerate(patterns):
            ax = axes[0][idx]
            pat_results = [r for r in results if r.pattern == pattern]

            if not pat_results:
                continue

            # Extract unique block counts and sizes
            block_counts = sorted(set(r.num_blocks for r in pat_results))
            sizes = sorted(set(r.size_mb for r in pat_results))

            # Build 2D bandwidth matrix
            bw_matrix = np.full((len(block_counts), len(sizes)), np.nan)
            for r in pat_results:
                bi = block_counts.index(r.num_blocks)
                si = sizes.index(r.size_mb)
                bw_matrix[bi, si] = r.bandwidth_gbps

            # Plot heatmap
            size_labels = [f"{s:.1f}" if s >= 1 else f"{s:.2f}" for s in sizes]
            block_labels = [str(b) for b in block_counts]

            # Use imshow for heatmap
            valid_bw = bw_matrix[~np.isnan(bw_matrix)]
            if len(valid_bw) == 0:
                continue

            vmin = max(valid_bw.min(), 1.0)
            vmax = valid_bw.max()

            im = ax.imshow(
                bw_matrix,
                aspect='auto',
                cmap='RdYlGn',
                interpolation='nearest',
                vmin=vmin,
                vmax=vmax,
            )

            # Axis labels
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(size_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(block_counts)))
            ax.set_yticklabels(block_labels, fontsize=8)
            ax.set_xlabel('Data Size (MB)')
            ax.set_ylabel('Block Count')
            ax.set_title(f'{pattern}\n(BW in GB/s)')

            # Mark L2 cache boundary with a vertical line
            l2_mb = self.l2_cache_mb
            for si, s in enumerate(sizes):
                if s >= l2_mb:
                    ax.axvline(x=si - 0.5, color='red', linewidth=2,
                               linestyle='--', label=f'L2={l2_mb:.0f}MB')
                    break

            # Annotate cells with bandwidth values
            for bi in range(len(block_counts)):
                for si in range(len(sizes)):
                    val = bw_matrix[bi, si]
                    if not np.isnan(val):
                        text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                        ax.text(si, bi, f'{val:.0f}',
                                ha='center', va='center',
                                fontsize=6, color=text_color)

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Bandwidth (GB/s)')

            # Add L2 boundary legend
            ax.legend(loc='upper right', fontsize=8)

        fig.suptitle(
            f'Memory Bandwidth Heatmap (Triton) | {self.device_name} | '
            f'L2 Cache: {self.l2_cache_mb:.0f} MB',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Heatmap saved to: {output_path}")

    def save_csv(self, results: List[MemBwResult], path: str):
        """Save benchmark results to CSV file."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'pattern', 'size_mb', 'dtype',
                    'num_blocks', 'block_size',
                    'median_time_ms', 'std_time_ms',
                    'bandwidth_gbps', 'peak_bandwidth_gbps', 'utilization_pct',
                    'in_l2_cache',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.pattern, r.size_mb, r.dtype,
                        r.num_blocks, r.block_size,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}",
                        f"{r.bandwidth_gbps:.4f}", f"{r.peak_bandwidth_gbps:.4f}",
                        f"{r.utilization*100:.2f}",
                        r.in_l2_cache,
                    ])
            print(f"[INFO] Results saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
