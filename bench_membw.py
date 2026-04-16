"""
GPU Memory Bandwidth benchmark using Triton kernels.

Measures effective memory bandwidth with fine-grained control over:
- Sequential (contiguous) memory access
- Strided (non-contiguous / scattered) memory access
- Triton autotune for optimal BLOCK_SIZE and num_warps selection
- Data sizes spanning L2 cache boundary

Key metrics:
- Effective bandwidth (GB/s)
- Bandwidth utilization vs theoretical peak

Generates curve plots: data_size -> bandwidth for each pattern,
highlighting L2 cache effects.
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .timing import bench_gpu_time
from .hw_spec import get_peak_bandwidth, get_l2_cache_size


# ===================================================================
# Autotune 配置
# ===================================================================

def _get_autotune_configs():
    """生成 autotune 候选配置：不同 BLOCK_SIZE 和 num_warps 组合。"""
    configs = []
    for block_size in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        for num_warps in [4, 8, 16]:
            # num_warps * 32 (warp_size) 不能超过 BLOCK_SIZE
            if num_warps * 32 <= block_size:
                configs.append(
                    triton.Config(
                        {'BLOCK_SIZE': block_size},
                        num_warps=num_warps,
                    )
                )
    return configs


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class MemBwResult:
    """Result of a single memory bandwidth benchmark run."""
    pattern: str          # e.g. 'seq_copy', 'seq_read', 'seq_write', 'strided_copy', 'strided_read'
    size_mb: float        # Buffer size in MB
    dtype: str
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
            f"pattern={self.pattern:<14} size={self.size_mb:8.2f}MB | "
            f"dtype={self.dtype:<8} | "
            f"time={self.median_time_ms:.3f}±{self.std_time_ms:.3f}ms | "
            f"BW={self.bandwidth_gbps:.1f}GB/s (util={util_str}) [{l2_str}]"
        )


# ===================================================================
# Triton kernels (with autotune)
# ===================================================================

@triton.autotune(configs=_get_autotune_configs(), key=['n_elements'])
@triton.jit
def _seq_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential copy: contiguous read from src + contiguous write to dst."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.autotune(configs=_get_autotune_configs(), key=['n_elements'])
@triton.jit
def _seq_read_kernel(
    src_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequential read: contiguous read, accumulate to prevent optimization."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(out_ptr, block_sum)


@triton.autotune(configs=_get_autotune_configs(), key=['n_elements'])
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


@triton.autotune(configs=_get_autotune_configs(), key=['n_elements'])
@triton.jit
def _strided_copy_kernel(
    src_ptr, dst_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided copy: gather read via indices + scatter write via indices."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = tl.load(indices_ptr + offsets, mask=mask)
    data = tl.load(src_ptr + idx, mask=mask)
    tl.store(dst_ptr + idx, data, mask=mask)


@triton.autotune(configs=_get_autotune_configs(), key=['n_elements'])
@triton.jit
def _strided_read_kernel(
    src_ptr, out_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided read: gather read via indices, reduce to scalar."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    idx = tl.load(indices_ptr + offsets, mask=mask)
    data = tl.load(src_ptr + idx, mask=mask)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(out_ptr, block_sum)


# ===================================================================
# Benchmark class
# ===================================================================

class MemBwBenchmark:
    """
    GPU Memory Bandwidth benchmark using Triton kernels with autotune.

    Tests sequential (contiguous) and strided (non-contiguous) memory access
    patterns with varying data sizes. Triton autotune automatically selects
    the best BLOCK_SIZE and num_warps for each configuration. Generates
    curve plots showing bandwidth as a function of data size for each pattern,
    highlighting L2 cache boundary effects.

    Example usage:
        bench = MemBwBenchmark()
        results = bench.run(
            sizes_mb=[1, 4, 16, 64, 256, 1024],
            patterns=['seq_copy', 'strided_copy'],
            dtypes=['float32'],
        )
        bench.plot(results, 'membw_curve.png')
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
        self.l2_cache_bytes = get_l2_cache_size(self.device_name)
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
        indices = torch.arange(n_elements, device=self.device, dtype=torch.int32)
        indices = (indices * stride) % n_elements
        return indices

    def run_single(
        self,
        size_mb: float,
        pattern: str = 'seq_copy',
        dtype_str: str = 'float32',
    ) -> Optional[MemBwResult]:
        """
        Run a single memory bandwidth benchmark with a Triton autotuned kernel.

        Triton autotune 会自动选择最佳的 BLOCK_SIZE 和 num_warps 配置。

        Args:
            size_mb: Buffer size in MB (per buffer).
            pattern: Access pattern. One of:
                     'seq_copy'     - Sequential copy: dst = src
                     'seq_read'     - Sequential read: sum(src)
                     'seq_write'    - Sequential write: dst = val
                     'strided_copy' - Strided copy (stride=128 elements)
                     'strided_read' - Strided read (stride=128 elements)
            dtype_str: Data type string.

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
        n_elements = max(1024, n_elements)

        # Stride for non-contiguous patterns
        stride = 128

        # Grid 由 autotune 选中的 BLOCK_SIZE 动态决定
        def grid(meta):
            return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

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
                    _seq_copy_kernel[grid](
                        src, dst, n_elements,
                    )
                bytes_transferred = 2 * n_elements * bpe  # read + write
            elif pattern == 'seq_read':
                def fn():
                    out_scalar.zero_()
                    _seq_read_kernel[grid](
                        src, out_scalar, n_elements,
                    )
                bytes_transferred = n_elements * bpe  # read only
            elif pattern == 'seq_write':
                def fn():
                    _seq_write_kernel[grid](
                        dst, n_elements,
                    )
                bytes_transferred = n_elements * bpe  # write only
            elif pattern == 'strided_copy':
                def fn():
                    _strided_copy_kernel[grid](
                        src, dst, indices, n_elements,
                    )
                bytes_transferred = 2 * n_elements * bpe  # read + write
            elif pattern == 'strided_read':
                def fn():
                    out_scalar.zero_()
                    _strided_read_kernel[grid](
                        src, out_scalar, indices, n_elements,
                    )
                bytes_transferred = n_elements * bpe  # read only
            else:
                print(f"[ERROR] Unknown pattern: {pattern}. "
                      f"Supported: seq_copy, seq_read, seq_write, strided_copy, strided_read")
                return None

            # Determine if data fits in L2 cache
            in_l2 = (n_elements * bpe) <= self.l2_cache_bytes

            # Flush L2 cache for HBM-bound tests to get realistic bandwidth
            median_ms, std_ms = bench_gpu_time(
                fn,
                enable_cupti=self.enable_cupti,
                num_iters=self.num_iters,
                dry_run_iters=self.dry_run_iters,
                cold_l2_cache=(not in_l2),
            )

        except Exception as e:
            print(f"[ERROR] MemBw failed pattern={pattern} size={size_mb}MB "
                  f"dtype={dtype_str}: {e}")
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
    ) -> List[MemBwResult]:
        """
        Run memory bandwidth benchmarks sweeping data sizes.

        Triton autotune 自动为每个 (pattern, size, dtype) 组合选择最佳的
        BLOCK_SIZE 和 num_warps 配置，无需手动指定 block_counts。

        Args:
            sizes_mb: Buffer sizes in MB. Defaults to auto-generated range
                      spanning L2 cache boundary.
            patterns: Access patterns. Defaults to ['seq_copy', 'strided_copy'].
            dtypes: Data types. Defaults to ['float32'].

        Returns:
            List of MemBwResult.
        """
        if sizes_mb is None:
            sizes_mb = self._get_default_sizes_mb()
        if patterns is None:
            patterns = ['seq_copy', 'strided_copy']
        if dtypes is None:
            dtypes = ['float32']

        results = []
        print(f"\n{'='*100}")
        print(f"Memory Bandwidth Benchmark (Triton Autotune) | Device: {self.device_name}")
        print(f"L2 Cache: {self.l2_cache_mb:.0f} MB | "
              f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"Data sizes (MB): {[f'{s:.2f}' for s in sizes_mb]}")
        print(f"Patterns: {patterns}")
        print(f"{'='*100}")
        print(f"{'pattern':<14} {'size(MB)':>8} | {'dtype':<8} | "
              f"{'time(ms)':>12} | {'BW(GB/s)':>10} | {'util':>7} | {'L2?':>4}")
        print(f"{'-'*100}")

        for dtype_str in dtypes:
            for pattern in patterns:
                for size_mb in sizes_mb:
                    result = self.run_single(size_mb, pattern, dtype_str)
                    if result is not None:
                        results.append(result)
                        util_str = (f"{result.utilization*100:.1f}%"
                                    if result.utilization > 0 else "N/A")
                        l2_str = "Y" if result.in_l2_cache else "N"
                        print(
                            f"{result.pattern:<14} "
                            f"{result.size_mb:>8.2f} | "
                            f"{result.dtype:<8} | "
                            f"{result.median_time_ms:>8.3f}±"
                            f"{result.std_time_ms:.3f} | "
                            f"{result.bandwidth_gbps:>10.1f} | "
                            f"{util_str:>7} | "
                            f"{l2_str:>4}"
                        )
                    else:
                        print(f"{pattern:<14} {size_mb:>8.2f} | "
                              f"{dtype_str:<8} | {'FAILED':>12}")

        print(f"{'='*100}")
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
                      f"(size={best_l2.size_mb:.2f}MB)")
            if hbm_results:
                best_hbm = max(hbm_results, key=lambda r: r.bandwidth_gbps)
                print(f"  Best BW (HBM)         : {best_hbm.bandwidth_gbps:.1f} GB/s "
                      f"(size={best_hbm.size_mb:.2f}MB)")
            if pat_results:
                best = max(pat_results, key=lambda r: r.bandwidth_gbps)
                print(f"  Peak BW (device)      : {best.peak_bandwidth_gbps:.1f} GB/s")
                if best.utilization > 0:
                    print(f"  Best utilization      : {best.utilization*100:.1f}%")

    def plot(
        self,
        results: List[MemBwResult],
        output_path: str = 'membw_curve.png',
        pattern_filter: List[str] = None,
    ):
        """
        在一张图上绘制不同 pattern 的 size-BW 曲线。

        X 轴为 data size (MB, log scale)，Y 轴为 bandwidth (GB/s)。
        用竖线标注 L2 cache 边界，用水平虚线标注理论峰值带宽。

        Args:
            results: List of MemBwResult from run().
            output_path: Path to save the figure.
            pattern_filter: If set, only plot these patterns.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping plot. "
                  "Install with: pip install matplotlib")
            return

        if not results:
            print("[WARNING] No results to plot.")
            return

        patterns = sorted(set(r.pattern for r in results))
        if pattern_filter:
            patterns = [p for p in patterns if p in pattern_filter]

        if not patterns:
            print("[WARNING] No matching patterns found for plotting.")
            return

        # 配色和标记样式
        style_map = {
            'seq_copy':     {'color': '#2196F3', 'marker': 'o', 'label': 'Sequential Copy (R+W)'},
            'seq_read':     {'color': '#4CAF50', 'marker': 's', 'label': 'Sequential Read'},
            'seq_write':    {'color': '#FF9800', 'marker': '^', 'label': 'Sequential Write'},
            'strided_copy': {'color': '#F44336', 'marker': 'D', 'label': 'Strided Copy (R+W)'},
            'strided_read': {'color': '#9C27B0', 'marker': 'v', 'label': 'Strided Read'},
        }
        # 默认样式（用于未知 pattern）
        default_colors = ['#00BCD4', '#795548', '#607D8B', '#E91E63', '#3F51B5']

        fig, ax = plt.subplots(figsize=(12, 7))

        peak_bw = 0.0
        for i, pattern in enumerate(patterns):
            pat_results = sorted(
                [r for r in results if r.pattern == pattern],
                key=lambda r: r.size_mb,
            )
            if not pat_results:
                continue

            sizes = [r.size_mb for r in pat_results]
            bws = [r.bandwidth_gbps for r in pat_results]
            peak_bw = max(peak_bw, pat_results[0].peak_bandwidth_gbps)

            style = style_map.get(pattern, {
                'color': default_colors[i % len(default_colors)],
                'marker': 'o',
                'label': pattern,
            })

            ax.plot(
                sizes, bws,
                color=style['color'],
                marker=style['marker'],
                markersize=6,
                linewidth=2,
                label=style['label'],
                alpha=0.9,
            )

        # 标注理论峰值带宽
        if peak_bw > 0:
            ax.axhline(
                y=peak_bw, color='gray', linewidth=1.5,
                linestyle='--', alpha=0.7,
                label=f'Peak BW ({peak_bw:.0f} GB/s)',
            )

        # 标注 L2 cache 边界
        l2_mb = self.l2_cache_mb
        ax.axvline(
            x=l2_mb, color='red', linewidth=1.5,
            linestyle=':', alpha=0.7,
            label=f'L2 Cache ({l2_mb:.0f} MB)',
        )

        # 在 L2 边界两侧添加区域标注
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], l2_mb, alpha=0.05, color='green')
        ax.axvspan(l2_mb, xlim[1], alpha=0.05, color='orange')
        ax.set_xlim(xlim)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Data Size (MB)', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(
            f'Memory Bandwidth vs Data Size | {self.device_name}\n'
            f'L2 Cache: {l2_mb:.0f} MB | Triton Autotune',
            fontsize=13, fontweight='bold',
        )
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=10)

        # X 轴刻度标签
        all_sizes = sorted(set(r.size_mb for r in results if r.pattern in patterns))
        if len(all_sizes) <= 20:
            ax.set_xticks(all_sizes)
            ax.set_xticklabels(
                [f"{s:.1f}" if s >= 1 else f"{s:.2f}" for s in all_sizes],
                rotation=45, ha='right', fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Curve plot saved to: {output_path}")

    # 保留旧方法名作为别名，兼容已有调用
    def plot_heatmap(self, results, output_path='membw_curve.png', pattern_filter=None):
        """Deprecated: 请使用 plot() 方法。此方法保留用于向后兼容。"""
        filter_list = [pattern_filter] if isinstance(pattern_filter, str) else pattern_filter
        return self.plot(results, output_path, pattern_filter=filter_list)

    def save_csv(self, results: List[MemBwResult], path: str):
        """Save benchmark results to CSV file."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'pattern', 'size_mb', 'dtype',
                    'median_time_ms', 'std_time_ms',
                    'bandwidth_gbps', 'peak_bandwidth_gbps', 'utilization_pct',
                    'in_l2_cache',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.pattern, r.size_mb, r.dtype,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}",
                        f"{r.bandwidth_gbps:.4f}", f"{r.peak_bandwidth_gbps:.4f}",
                        f"{r.utilization*100:.2f}",
                        r.in_l2_cache,
                    ])
            print(f"[INFO] Results saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
