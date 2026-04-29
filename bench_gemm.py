"""
GEMM (General Matrix Multiply) benchmark for GPU.

Measures:
- Execution time (ms)
- TFLOPS throughput
- MFU (Model Flops Utilization) relative to theoretical peak
- Memory bandwidth (GB/s)

Supports dtypes: float32, float16, bfloat16, int8, float8_e4m3fn, float4_e2m1fn_x2
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .timing import bench_gpu_time
from . import xpu_device as xpu
from .hw_spec import (
    DTYPE_FROM_STR, DTYPE_OUTPUT_MAPPING, DTYPE_BYTES,
    get_peak_tflops, get_peak_bandwidth,
)


# ===================================================================
# LLM Model Shape
# ===================================================================

# Pre-defined LLM model configurations.
# 每个条目: (workload_name, [K, N], split_dim)
#   K          : 输入维度
#   N          : 输出维度
#   split_dim  : 张量并行切分维度
#                1 -> 列并行, 沿 N 切分 (e.g. QKV / Moe_gate_up)
#                0 -> 行并行, 沿 K 切分 (e.g. Proj / Moe_down)
MODEL_SHAPE: Dict[str, List] = {
    "Basic": [
        ('Mx4096x4096', [4096, 4096], 1),
    ],
    "HY-image-3.0": [
        ('QKV', [4096, 6144], 1),
        ('Proj', [4096, 4096], 0),
        ('Moe_gate_up', [4096, 6144], 1),
        ('Moe_down', [3072, 4096], 0),
    ],
    "DeepSeek-V3": [
        ('QKV_Lora', [2048, 7168], 1),
        ('QK_Lora_b', [2048, 2624], 0),
        ('V_Lora_b', [2048, 21888], 1),
        ('Proj', [10944, 2048], 0),
        ('Moe_gate_up', [7168, 4096], 1),
        ('Moe_down', [2048, 7168], 0),
    ],
}


@dataclass
class GemmResult:
    """Result of a single GEMM benchmark run.

    通用 GEMM 结果. 当该 GEMM 是某个 LLM workload 的一部分时,
    额外填充 model_name / workload_name / batch_size / tp 字段.
    """
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
    mbu: float               # Bandwidth utilization (0~1)
    device_name: str
    # ---- optional LLM workload metadata ----
    model_name: str = ''
    workload_name: str = ''
    batch_size: int = 0
    tp: int = 1


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


class GemmBenchmark:
    """
    GPU GEMM performance benchmark.

    既支持 naive shape 的 GEMM 测试 (run_single), 也支持基于 LLM 模型结构
    (QKV / Proj / MoE 等) 的批量测试 (run).

    Example usage:
        bench = GemmBenchmark()

        # 单个 shape
        r = bench.run_single(4096, 4096, 4096, 'bfloat16')

        # LLM workload sweep
        results = bench.run(
            model_name='HY-image-3.0',
            batch_sizes=[1, 4, 16, 64, 256, 1024, 4096],
            dtypes=['bfloat16'],
            tp=1,
        )
        bench.print_summary(results)
        bench.save_csv(results, 'gemm_results.csv')
        bench.plot_batch_tflops_curve(results, 'gemm_curves.png')
    """

    def __init__(
        self,
        device: str = None,
        num_iters: int = 30,
        dry_run_iters: int = 5,
        enable_cupti: bool = False,
    ):
        if not xpu.is_available():
            raise RuntimeError("No XPU (CUDA / NPU) device available.")
        self.device = device if device is not None else xpu.default_device_str()
        self.num_iters = num_iters
        self.dry_run_iters = dry_run_iters
        self.enable_cupti = enable_cupti
        self.device_name = xpu.get_device_name(0)

    # ------------------------------------------------------------------
    # Tensor / kernel helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_tensors(
        m: int, n: int, k: int,
        dtype: torch.dtype,
        device: str,
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
            # NOTE: torch.randint / torch.randn 等随机算子不支持直接生成 float8_e4m3fn,
            # cuBLASLt 对 FP8 GEMM 的约束: A 必须 row-major, B 必须 column-major.
            # 做法: 先按 (N, K) 行主序分配, 再 .t() 得到 shape=(K, N) 的列主序视图
            # (stride 不连续, 不要 .contiguous()).
            a = torch.randn((m, k), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
            b = torch.randn((n, k), device=device, dtype=torch.float32).to(torch.float8_e4m3fn).t()
        else:
            # TODO: support torch.float4_e2m1fn_x2
            raise ValueError(f"Unsupported dtype: {dtype}")

        return a, b

    @staticmethod
    def _run_gemm_kernel(
        a: torch.Tensor,
        b: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute GEMM for the given dtype."""
        # TODO: support FP4 gemm, cutlass, triton, TileLang
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            return torch.matmul(a, b)
        elif dtype == torch.int8:
            # Use torch._int_mm for int8 (returns int32). NPU 可能不支持。
            return torch._int_mm(a, b)
        elif dtype == torch.float8_e4m3fn:
            # Use torch._scaled_mm for FP8 (CUDA only)
            scale_a = torch.tensor(1.0, device=a.device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=b.device, dtype=torch.float32)
            result_tuple = torch._scaled_mm(
                a, b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            return result_tuple[0]
        else:
            raise ValueError(f"Unsupported dtype for GEMM: {dtype}")

    # ------------------------------------------------------------------
    # Single GEMM
    # ------------------------------------------------------------------
    def run_single(
        self,
        m: int,
        n: int,
        k: int,
        dtype_str: str = 'bfloat16',
        *,
        model_name: str = '',
        workload_name: str = '',
        batch_size: int = 0,
        tp: int = 1,
    ) -> Optional[GemmResult]:
        """
        Run a single GEMM benchmark.

        Args:
            m, n, k: Matrix dimensions. C(m,n) = A(m,k) @ B(k,n)
            dtype_str: Data type string, e.g. 'bfloat16', 'float16', 'float32',
                       'int8', 'float8_e4m3fn'
            model_name / workload_name / batch_size / tp: 可选的 LLM workload 元信息.

        Returns:
            GemmResult or None on failure.
        """
        if dtype_str not in DTYPE_FROM_STR:
            print(f"[ERROR] Unsupported dtype: {dtype_str}")
            return None

        input_dtype = DTYPE_FROM_STR[dtype_str]
        output_dtype = DTYPE_OUTPUT_MAPPING[input_dtype]

        try:
            a, b = self._create_tensors(m, n, k, input_dtype, self.device)

            def fn():
                return self._run_gemm_kernel(a, b, input_dtype)

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
        theory_time_ms = max(
            (flops / hw_tflops / 1e9) if hw_tflops > 0 else 0.0,
            (total_bytes / hw_bandwidth / 1e6) if hw_bandwidth > 0 else 0.0,
        )  # ms
        theory_tflops = (flops / 1e9) / theory_time_ms if theory_time_ms > 0 else 0.0

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
            model_name=model_name,
            workload_name=workload_name,
            batch_size=batch_size,
            tp=tp,
        )

    # ------------------------------------------------------------------
    # LLM workload sweep
    # ------------------------------------------------------------------
    def run(
        self,
        model_name: str = 'HY-image-3.0',
        batch_sizes: List[int] = None,
        dtypes: List[str] = None,
        tp: int = 1,
    ) -> List[GemmResult]:
        """
        Run LLM GEMM benchmarks across batch sizes.

        直接从 MODEL_SHAPE 中读取 (name, [K, N], split_dim), 按 tp 切分后
        对每个 batch_size 调用 run_single.

        TP 切分规则:
            split_dim == 1 (列并行, e.g. QKV / Moe_gate_up): N -> N // tp
            split_dim == 0 (行并行, e.g. Proj   / Moe_down ): K -> K // tp

        Args:
            model_name: Key in MODEL_SHAPE dict (e.g. 'HY-image-3.0', 'DeepSeek-V3').
            batch_sizes: List of batch sizes (= num tokens) to sweep.
                         Defaults to [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].
            dtypes: Data types to test. Defaults to ['bfloat16'].
            tp: Tensor parallelism degree.

        Returns:
            List of GemmResult (with LLM workload metadata filled).
        """
        if model_name not in MODEL_SHAPE:
            available = ', '.join(MODEL_SHAPE.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        shape_list = MODEL_SHAPE[model_name]

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if dtypes is None:
            dtypes = ['bfloat16']

        results: List[GemmResult] = []

        print(f"\n{'='*120}")
        print(f"LLM GEMM Benchmark | Model: {model_name} | Device: {self.device_name} | TP={tp}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Dtypes: {dtypes}")
        print(f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"{'='*120}")

        for dtype_str in dtypes:
            print(f"\n--- dtype: {dtype_str} ---")
            print(f"{'workload':<14} {'batch':>6} | {'M':>6} {'N':>6} {'K':>6} | "
                  f"{'time(ms)':>12} | {'TFLOPS':>8} | {'theory_TFLOPS':>8} | {'MFU':>4} | "
                  f"{'BW(GB/s)':>10} | {'MBU':>7}")
            print(f"{'-'*120}")

            for batch_size in batch_sizes:
                for wl_name, (k_base, n_base), split_dim in shape_list:
                    # 按 tp 切分
                    if split_dim == 1:
                        # 列并行: 切 N
                        k = k_base
                        n = n_base // tp
                    elif split_dim == 0:
                        # 行并行: 切 K
                        k = k_base // tp
                        n = n_base
                    else:
                        raise ValueError(
                            f"Unsupported split_dim={split_dim} for workload '{wl_name}'"
                        )

                    m = batch_size
                    if (wl_name == 'Moe_gate_up' or wl_name == 'Moe_down'):
                        # TODO: m = max(batch_size * topk // num_experts, 1)
                        m = max(batch_size // 8, 1)
                    if (dtype_str == 'int8') and (m <= 16):
                        continue

                    result = self.run_single(
                        m, n, k, dtype_str,
                        model_name=model_name,
                        workload_name=wl_name,
                        batch_size=batch_size,
                        tp=tp,
                    )

                    if result is not None:
                        results.append(result)

                        mfu_str = f"{result.mfu*100:.1f}%" if result.mfu > 0 else "N/A"
                        mbu_str = f"{result.mbu*100:.1f}%" if result.mbu > 0 else "N/A"
                        print(
                            f"{wl_name:<14} {batch_size:>6} | "
                            f"{m:>6} {n:>6} {k:>6} | "
                            f"{result.median_time_ms:>8.3f}±{result.std_time_ms:.3f} | "
                            f"{result.tflops:>8.2f} | "
                            f"{result.theory_tflops:>8.2f} | "
                            f"{mfu_str:>7} | "
                            f"{result.bandwidth_gbps:>10.1f} | "
                            f"{mbu_str:>7}"
                        )
                    else:
                        print(f"{wl_name:<14} {batch_size:>6} | "
                              f"{m:>6} {n:>6} {k:>6} | {'FAILED':>12}")

        print(f"{'='*120}")
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_summary(self, results: List[GemmResult]):
        """Print summary grouped by workload type."""
        if not results:
            print("No results to summarize.")
            return

        model_name = results[0].model_name or 'N/A'
        print(f"\n{'='*80}")
        print(f"GEMM Benchmark Summary | Model: {model_name}")
        print(f"{'='*80}")

        # Group by workload name
        workload_names = sorted(set(r.workload_name for r in results))
        for wl_name in workload_names:
            wl_results = [r for r in results if r.workload_name == wl_name]
            best = max(wl_results, key=lambda r: r.tflops)
            worst = min(wl_results, key=lambda r: r.tflops)
            print(f"  {wl_name:<14}: "
                  f"best={best.tflops:.2f} TFLOPS (batch={best.batch_size}) MFU={best.mfu*100:.1f}% | "
                  f"worst={worst.tflops:.2f} TFLOPS (batch={worst.batch_size}) MBU={worst.mbu*100:.1f}%")

    def save_csv(self, results: List[GemmResult], path: str):
        """Save GEMM benchmark results to CSV file."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'model', 'workload',
                    'batch_size', 'tp', 'dtype',
                    'M', 'N', 'K',
                    'median_time_ms', 'std_time_ms', 'tflops',
                    'theory_tflops', 'mfu_pct',
                    'bandwidth_gbps', 'mbu_pct',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.model_name, r.workload_name,
                        r.batch_size, r.tp, r.dtype,
                        r.m, r.n, r.k,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}",
                        f"{r.tflops:.4f}", f"{r.theory_tflops:.4f}",
                        f"{r.mfu*100:.2f}",
                        f"{r.bandwidth_gbps:.4f}", f"{r.mbu*100:.2f}",
                    ])
            print(f"[INFO] GEMM results saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    def plot_batch_tflops_curve(
        self,
        results: List[GemmResult],
        output_path: str = 'llm_gemm_combined_tflops.png',
    ):
        """
        在一张图中绘制多个 workload × 多种 dtype 的 batch_size vs TFLOPS 曲线。

        编码方式:
            - 颜色 + marker : 区分不同的 workload (pattern), e.g. QKV / Proj / Moe_gate_up ...
            - linestyle     : 区分不同的 dtype,            e.g. bf16 / fp16 / fp8 / int8
            - 水平虚线      : 各 dtype 的硬件 TFLOPS 峰值 (linestyle 与该 dtype 对齐)

        Args:
            results: List of GemmResult from run().
            output_path: 输出图片路径 (单文件, 不再按 dtype 拆分)。
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.ticker import ScalarFormatter
        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping plot. "
                  "Install with: pip install matplotlib")
            return

        if not results:
            print("[WARNING] No results to plot.")
            return

        model_name = results[0].model_name or 'N/A'

        # 保持首次出现顺序: workloads 和 dtypes
        workload_filter = []
        _seen_wl = set()
        for r in results:
            if r.workload_name not in _seen_wl:
                workload_filter.append(r.workload_name)
                _seen_wl.add(r.workload_name)

        dtypes = []
        _seen_dt = set()
        for r in results:
            if r.dtype not in _seen_dt:
                dtypes.append(r.dtype)
                _seen_dt.add(r.dtype)

        # workload -> (color, marker)
        style_map = {
            'QKV':          {'color': '#2196F3', 'marker': 'o'},
            'Proj':         {'color': '#4CAF50', 'marker': 's'},
            'Moe_gate_up':  {'color': '#FF9800', 'marker': 'D'},
            'Moe_down':     {'color': '#9C27B0', 'marker': '^'},
            'QKV_Lora':     {'color': '#2196F3', 'marker': 'o'},
            'QK_Lora_b':    {'color': '#03A9F4', 'marker': 'v'},
            'V_Lora_b':     {'color': '#009688', 'marker': '<'},
        }
        fallback_colors = ['#E91E63', '#00BCD4', '#795548', '#607D8B']
        fallback_markers = ['v', '<', '>', 'p']

        # dtype -> linestyle
        linestyle_pool = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        dtype_linestyle = {
            dt: linestyle_pool[i % len(linestyle_pool)]
            for i, dt in enumerate(dtypes)
        }

        # 分配 workload 样式 (对 fallback 动态编号)
        wl_style = {}
        fallback_idx = 0
        for wl_name in workload_filter:
            if wl_name in style_map:
                wl_style[wl_name] = dict(style_map[wl_name])
            else:
                wl_style[wl_name] = {
                    'color': fallback_colors[fallback_idx % len(fallback_colors)],
                    'marker': fallback_markers[fallback_idx % len(fallback_markers)],
                }
                fallback_idx += 1

        single_dtype = (len(dtypes) == 1)

        if single_dtype:
            # ========== 单 dtype: 沿用原始单图例布局 ==========
            dtype_str = dtypes[0]
            fig, ax = plt.subplots(figsize=(12, 7))

            for wl_name in workload_filter:
                style = wl_style[wl_name]
                wl_results = sorted(
                    [r for r in results
                     if r.workload_name == wl_name and r.dtype == dtype_str],
                    key=lambda r: r.batch_size,
                )
                if not wl_results:
                    continue

                batch_sizes = [r.batch_size for r in wl_results]
                tflops_vals = [r.tflops for r in wl_results]
                mfu_vals = [r.mfu * 100 for r in wl_results]

                ax.plot(
                    batch_sizes, tflops_vals,
                    marker=style['marker'], color=style['color'],
                    linewidth=2.2, markersize=7,
                    label=wl_name, zorder=3,
                )

                # 关键点上标注 MFU, 避免拥挤
                for i, (bs, tf, mfu) in enumerate(zip(batch_sizes, tflops_vals, mfu_vals)):
                    if i % max(1, len(batch_sizes) // 5) == 0 or i == len(batch_sizes) - 1:
                        ax.annotate(
                            f'{mfu:.0f}%',
                            (bs, tf),
                            textcoords="offset points",
                            xytext=(0, 10),
                            fontsize=7, ha='center',
                            color=style['color'], alpha=0.8,
                        )

            # HW Peak 水平线
            hw_tflops = get_peak_tflops(self.device_name, dtype_str)
            if hw_tflops > 0:
                ax.axhline(
                    y=hw_tflops, color='red', linewidth=2,
                    linestyle='--', alpha=0.7,
                    label=f'HW Peak ({hw_tflops:.0f} TFLOPS)',
                    zorder=2,
                )

            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel('Batch Size (tokens)', fontsize=12)
            ax.set_ylabel('TFLOPS', fontsize=12)
            ax.set_title(
                f'LLM GEMM Benchmark: {model_name} | {dtype_str} | {self.device_name}',
                fontsize=13, fontweight='bold',
            )
            ax.legend(fontsize=10, loc='center left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            ax.set_ylim(bottom=0)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Combined batch-TFLOPS plot saved to: {output_path}")
            return

        # ========== 多 dtype: 颜色/marker 区分 workload, linestyle 区分 dtype ==========
        fig, ax = plt.subplots(figsize=(13, 7.5))

        # 绘制每条 (workload, dtype) 曲线
        for wl_name in workload_filter:
            style = wl_style[wl_name]
            for dtype_str in dtypes:
                wl_results = sorted(
                    [r for r in results
                     if r.workload_name == wl_name and r.dtype == dtype_str],
                    key=lambda r: r.batch_size,
                )
                if not wl_results:
                    continue

                batch_sizes = [r.batch_size for r in wl_results]
                tflops_vals = [r.tflops for r in wl_results]

                ax.plot(
                    batch_sizes, tflops_vals,
                    marker=style['marker'],
                    color=style['color'],
                    linestyle=dtype_linestyle[dtype_str],
                    linewidth=2.0, markersize=6,
                    zorder=3,
                )

        # 为每个 dtype 画一条 HW Peak 水平线 (颜色固定红, linestyle 对齐该 dtype)
        peak_handles = []
        for dtype_str in dtypes:
            hw_tflops = get_peak_tflops(self.device_name, dtype_str)
            if hw_tflops > 0:
                ax.axhline(
                    y=hw_tflops, color='red', linewidth=1.6,
                    linestyle=dtype_linestyle[dtype_str], alpha=0.65,
                    zorder=2,
                )
                peak_handles.append(Line2D(
                    [0], [0], color='red', linewidth=1.6,
                    linestyle=dtype_linestyle[dtype_str], alpha=0.8,
                    label=f'HW Peak {dtype_str} ({hw_tflops:.0f} TFLOPS)',
                ))

        # ===== 构造双 legend: workload (颜色/marker) + dtype (linestyle) =====
        workload_handles = [
            Line2D(
                [0], [0],
                color=wl_style[wl]['color'],
                marker=wl_style[wl]['marker'],
                linestyle='-',
                linewidth=2.0, markersize=7,
                label=wl,
            )
            for wl in workload_filter
        ]
        dtype_handles = [
            Line2D(
                [0], [0],
                color='black',
                linestyle=dtype_linestyle[dt],
                linewidth=2.0,
                label=dt,
            )
            for dt in dtypes
        ]

        leg1 = ax.legend(
            handles=workload_handles,
            title='Workload',
            fontsize=9, title_fontsize=10,
            loc='upper left', bbox_to_anchor=(1.01, 1.0),
            frameon=True,
        )
        leg2 = ax.legend(
            handles=dtype_handles,
            title='Dtype (linestyle)',
            fontsize=9, title_fontsize=10,
            loc='upper left', bbox_to_anchor=(1.01, 0.55),
            frameon=True,
        )
        ax.add_artist(leg1)
        if peak_handles:
            leg3 = ax.legend(
                handles=peak_handles,
                title='HW Peak',
                fontsize=8, title_fontsize=9,
                loc='upper left', bbox_to_anchor=(1.01, 0.25),
                frameon=True,
            )
            ax.add_artist(leg2)

        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Batch Size (tokens)', fontsize=12)
        ax.set_ylabel('TFLOPS', fontsize=12)
        ax.set_title(
            f'LLM GEMM Benchmark: {model_name} | {self.device_name}',
            fontsize=13, fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Combined batch-TFLOPS plot saved to: {output_path}")
