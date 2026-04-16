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


# ===================================================================
# LLM Model Configurations
# ===================================================================

@dataclass
class LLMModelConfig:
    """Configuration for an LLM model architecture."""
    name: str
    hidden_size: int           # d_model / hidden dimension
    num_attention_heads: int   # number of attention heads
    num_kv_heads: int          # number of KV heads (GQA)
    shared_inter_size: int     # FFN intermediate dimension (dense models)
    num_experts: int = 0       # total number of experts
    top_k_experts: int = 0     # number of active experts per token
    moe_inter_size: int = 0  # per-expert FFN intermediate dimension
    vocab_size: int = 0        # vocabulary size (for lm_head)
    real_head_dim: int = 128         # explicit head_dim override (0 = auto from hidden_size/num_heads)

    @property
    def head_dim(self) -> int:
        if self.real_head_dim > 0:
            return self.real_head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def q_proj_size(self) -> int:
        """Q projection output dimension: num_attention_heads * head_dim."""
        return self.num_attention_heads * self.head_dim

    @property
    def k_proj_size(self) -> int:
        """K projection output dimension (GQA-aware)."""
        return self.num_kv_heads * self.head_dim

    @property
    def v_proj_size(self) -> int:
        """V projection output dimension (GQA-aware)."""
        return self.num_kv_heads * self.head_dim


# Pre-defined LLM model configurations
LLM_MODELS: Dict[str, LLMModelConfig] = {
    "HY-image-3.0": LLMModelConfig(
        name="HY-image-3.0",
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        shared_inter_size=3072,   # share expert
        num_experts=64,
        top_k_experts=8,
        moe_inter_size=3072,  # per-expert FFN intermediate
        vocab_size=129280,
    ),
    "deepseek-v3": LLMModelConfig(
        name="DeepSeek-V3",
        hidden_size=7168,
        num_attention_heads=128,
        num_kv_heads=128,
        shared_inter_size=18432,     # dense FFN (first 3 layers)
        num_experts=256,
        top_k_experts=8,
        moe_inter_size=2048,  # per-expert FFN intermediate
        vocab_size=129280,
    ),
    "qwen2.5-72b": LLMModelConfig(
        name="Qwen2.5-72B",
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        shared_inter_size=29568,
        vocab_size=152064,
    ),
    "qwen3-235b-a22b": LLMModelConfig(
        name="Qwen3-235B-A22B",
        hidden_size=4096,
        num_attention_heads=64,
        num_kv_heads=4,
        shared_inter_size=12288,        # dense FFN intermediate
        num_experts=128,
        top_k_experts=8,
        moe_inter_size=1536,     # per-expert FFN intermediate
        vocab_size=151936,
        real_head_dim=128,                  # explicit head_dim (hidden_size/num_heads=64, but actual=128)
    ),
}


@dataclass
class LLMGemmWorkload:
    """A single GEMM workload derived from an LLM layer."""
    name: str          # e.g. "QKV_Proj", "O_Proj", "Gate_Up", "Down", "MoE_Gate_Up", ...
    m: int             # batch_size * seq_len (or tokens routed to expert)
    n: int             # output dimension
    k: int             # input dimension
    description: str   # human-readable description
    category: str      # "attention" or "ffn" or "moe"


def get_llm_gemm_workloads(
    model: LLMModelConfig,
    batch_size: int,
    tp: int = 1,
) -> List[LLMGemmWorkload]:
    """
    Generate GEMM workloads for a given LLM model and batch_size (= num_tokens).

    Args:
        model: LLM model configuration.
        batch_size: Number of tokens (M dimension for GEMM).
        tp: Tensor parallelism degree (divides output dims).

    Returns:
        List of LLMGemmWorkload representing all GEMM ops in one transformer layer.
    """
    h = model.hidden_size
    workloads = []

    # --- Attention projections ---
    # QKV fused projection: [batch, hidden] -> [batch, q_size + 2*kv_size]
    qkv_out = (model.q_proj_size + model.k_proj_size + model.v_proj_size) // tp
    workloads.append(LLMGemmWorkload(
        name="QKV_Proj",
        m=batch_size, n=qkv_out, k=h,
        description=f"Fused QKV projection ({model.num_attention_heads} q_head, {model.num_kv_heads} kv_head)",
        category="attention",
    ))

    # Output projection: [batch, hidden] -> [batch, hidden]
    workloads.append(LLMGemmWorkload(
        name="O_Proj",
        m=batch_size, n=h // tp, k=h // tp,
        description="Attention output projection",
        category="attention",
    ))

    # --- FFN / MoE ---
    if model.num_experts > 0:
        # MoE gate: [batch, hidden] -> [batch, num_experts] (small GEMM)
        # workloads.append(LLMGemmWorkload(
        #     name="MoE_Gate",
        #     m=batch_size, n=model.num_experts, k=h,
        #     description=f"MoE router gate ({model.num_experts} experts)",
        #     category="moe",
        # ))

        # Per-expert gate+up fused: tokens_per_expert x (2*moe_inter) x hidden
        # Approximate tokens per expert = batch_size * top_k / num_experts
        # TODO: fuse_moe_kernel();
        tokens_per_expert = max(1, (batch_size * model.top_k_experts) // model.num_experts)
        moe_inter = model.moe_inter_size // tp

        workloads.append(LLMGemmWorkload(
            name="MoE_Gate_Up",
            m=tokens_per_expert, n=2 * moe_inter, k=h,
            description=f"MoE expert gate+up (top{model.top_k_experts}/{model.num_experts}, ~{tokens_per_expert} tokens/expert)",
            category="moe",
        ))

        workloads.append(LLMGemmWorkload(
            name="MoE_Down",
            m=tokens_per_expert, n=h, k=moe_inter,
            description=f"MoE expert down projection (~{tokens_per_expert} tokens/expert)",
            category="moe",
        ))
    else:
        # Dense FFN: gate+up fused [batch, hidden] -> [batch, 2*inter]
        inter = model.shared_inter_size // tp
        workloads.append(LLMGemmWorkload(
            name="Shared_Gate_Up",
            m=batch_size, n=2 * inter, k=h,
            description="FFN gate+up fused projection",
            category="ffn",
        ))

        # Down projection: [batch, inter] -> [batch, hidden]
        workloads.append(LLMGemmWorkload(
            name="Shared_Down",
            m=batch_size, n=h // tp, k=inter,
            description="FFN down projection",
            category="ffn",
        ))

    return workloads


@dataclass
class LLMGemmResult:
    """Result of an LLM GEMM workload benchmark."""
    model_name: str
    workload_name: str
    category: str
    description: str
    batch_size: int
    m: int
    n: int
    k: int
    dtype: str
    tp: int
    median_time_ms: float
    std_time_ms: float
    tflops: float
    theory_tflops: float
    mfu: float
    bandwidth_gbps: float
    mbu: float
    device_name: str

    def __str__(self) -> str:
        mfu_str = f"{self.mfu * 100:.1f}%" if self.mfu > 0 else "N/A"
        return (
            f"[{self.model_name}] {self.workload_name:<14} batch={self.batch_size:>6} | "
            f"M={self.m:>6} N={self.n:>6} K={self.k:>6} | "
            f"dtype={self.dtype:<10} | "
            f"time={self.median_time_ms:.3f}±{self.std_time_ms:.3f}ms | "
            f"TFLOPS={self.tflops:.2f} (MFU={mfu_str})"
        )


class LLMGemmBenchmark:
    """
    Benchmark GEMM workloads derived from LLM model architectures.

    Sweeps across batch_sizes and measures QKV, Output Projection, FFN,
    and MoE GEMM performance. Generates batch_size vs TFLOPS curves and CSV output.

    Example usage:
        bench = LLMGemmBenchmark()
        results = bench.run(
            model_name='deepseek-v3',
            batch_sizes=[1, 4, 16, 64, 256, 1024, 4096],
            dtypes=['bfloat16'],
        )
        bench.plot_batch_tflops(results, 'llm_gemm_curves.png')
        bench.save_csv(results, 'llm_gemm_results.csv')
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
        self._gemm_bench = GemmBenchmark(
            device=device,
            num_iters=num_iters,
            dry_run_iters=dry_run_iters,
            enable_cupti=enable_cupti,
        )

    def run(
        self,
        model_name: str = 'deepseek-v3',
        batch_sizes: List[int] = None,
        dtypes: List[str] = None,
        tp: int = 1,
    ) -> List[LLMGemmResult]:
        """
        Run LLM GEMM benchmarks across batch sizes.

        Args:
            model_name: Key in LLM_MODELS dict (e.g. 'deepseek-v3', 'qwen2.5-72b').
            batch_sizes: List of batch sizes (= num tokens) to sweep.
                         Defaults to [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].
            dtypes: Data types to test. Defaults to ['bfloat16'].
            tp: Tensor parallelism degree.

        Returns:
            List of LLMGemmResult.
        """
        if model_name not in LLM_MODELS:
            available = ', '.join(LLM_MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        model = LLM_MODELS[model_name]

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if dtypes is None:
            dtypes = ['bfloat16']

        results: List[LLMGemmResult] = []

        print(f"\n{'='*120}")
        print(f"LLM GEMM Benchmark | Model: {model.name} | Device: {self.device_name} | TP={tp}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Dtypes: {dtypes}")
        print(f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"{'='*120}")

        for dtype_str in dtypes:
            print(f"\n--- dtype: {dtype_str} ---")
            print(f"{'workload':<14} {'batch':>6} | {'M':>6} {'N':>6} {'K':>6} | "
                  f"{'time(ms)':>12} | {'TFLOPS':>8} | {'theory':>8} | {'MFU':>7} | "
                  f"{'BW(GB/s)':>10} | {'MBU':>7}")
            print(f"{'-'*120}")

            for batch_size in batch_sizes:
                workloads = get_llm_gemm_workloads(model, batch_size, tp=tp)

                for wl in workloads:
                    gemm_result = self._gemm_bench.run_single(
                        wl.m, wl.n, wl.k, dtype_str,
                    )

                    if gemm_result is not None:
                        llm_result = LLMGemmResult(
                            model_name=model.name,
                            workload_name=wl.name,
                            category=wl.category,
                            description=wl.description,
                            batch_size=batch_size,
                            m=wl.m, n=wl.n, k=wl.k,
                            dtype=dtype_str,
                            tp=tp,
                            median_time_ms=gemm_result.median_time_ms,
                            std_time_ms=gemm_result.std_time_ms,
                            tflops=gemm_result.tflops,
                            theory_tflops=gemm_result.theory_tflops,
                            mfu=gemm_result.mfu,
                            bandwidth_gbps=gemm_result.bandwidth_gbps,
                            mbu=gemm_result.mbu,
                            device_name=gemm_result.device_name,
                        )
                        results.append(llm_result)

                        mfu_str = f"{llm_result.mfu*100:.1f}%" if llm_result.mfu > 0 else "N/A"
                        mbu_str = f"{llm_result.mbu*100:.1f}%" if llm_result.mbu > 0 else "N/A"
                        print(
                            f"{wl.name:<14} {batch_size:>6} | "
                            f"{wl.m:>6} {wl.n:>6} {wl.k:>6} | "
                            f"{gemm_result.median_time_ms:>8.3f}±{gemm_result.std_time_ms:.3f} | "
                            f"{gemm_result.tflops:>8.2f} | "
                            f"{gemm_result.theory_tflops:>8.2f} | "
                            f"{mfu_str:>7} | "
                            f"{gemm_result.bandwidth_gbps:>10.1f} | "
                            f"{mbu_str:>7}"
                        )
                    else:
                        print(f"{wl.name:<14} {batch_size:>6} | "
                              f"{wl.m:>6} {wl.n:>6} {wl.k:>6} | {'FAILED':>12}")

        print(f"{'='*120}")
        return results

    def print_summary(self, results: List[LLMGemmResult]):
        """Print summary grouped by workload type."""
        if not results:
            print("No results to summarize.")
            return

        model_name = results[0].model_name
        print(f"\n{'='*80}")
        print(f"LLM GEMM Summary | Model: {model_name}")
        print(f"{'='*80}")

        # Group by category then workload name
        categories = ['attention', 'ffn', 'moe']
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue

            print(f"\n[{cat.upper()}]")
            workload_names = sorted(set(r.workload_name for r in cat_results))
            for wl_name in workload_names:
                wl_results = [r for r in cat_results if r.workload_name == wl_name]
                best = max(wl_results, key=lambda r: r.tflops)
                worst = min(wl_results, key=lambda r: r.tflops)
                print(f"  {wl_name:<14}: "
                      f"best={best.tflops:.2f} TFLOPS (batch={best.batch_size}) | "
                      f"worst={worst.tflops:.2f} TFLOPS (batch={worst.batch_size}) | "
                      f"best MFU={best.mfu*100:.1f}%")

    def plot_batch_tflops(
        self,
        results: List[LLMGemmResult],
        output_path: str = 'llm_gemm_batch_tflops.png',
        dtype_filter: str = None,
    ):
        """
        Plot batch_size vs TFLOPS curves for each workload type.

        Generates a multi-panel figure with one subplot per workload,
        showing how TFLOPS scales with batch_size. Includes theoretical
        peak line for reference.

        Args:
            results: List of LLMGemmResult from run().
            output_path: Path to save the figure.
            dtype_filter: If set, only plot results for this dtype.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.ticker import ScalarFormatter
        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping plot. "
                  "Install with: pip install matplotlib")
            return

        if dtype_filter:
            results = [r for r in results if r.dtype == dtype_filter]

        if not results:
            print("[WARNING] No results to plot.")
            return

        model_name = results[0].model_name

        # Group by dtype
        dtypes = sorted(set(r.dtype for r in results))

        for dtype_str in dtypes:
            dtype_results = [r for r in results if r.dtype == dtype_str]

            # Get unique workload names preserving order
            seen = set()
            workload_names = []
            for r in dtype_results:
                if r.workload_name not in seen:
                    seen.add(r.workload_name)
                    workload_names.append(r.workload_name)

            n_workloads = len(workload_names)
            if n_workloads == 0:
                continue

            # Layout: up to 3 columns
            n_cols = min(3, n_workloads)
            n_rows = (n_workloads + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(7 * n_cols, 5 * n_rows),
                squeeze=False,
            )

            # Category -> color mapping
            category_colors = {
                'attention': '#2196F3',  # blue
                'ffn': '#4CAF50',        # green
                'moe': '#FF9800',        # orange
            }
            category_markers = {
                'attention': 'o',
                'ffn': 's',
                'moe': 'D',
            }

            # Get theoretical peak for reference
            hw_tflops = 0.0
            if dtype_results:
                hw_tflops = get_peak_tflops(self.device_name, dtype_str)

            for idx, wl_name in enumerate(workload_names):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]

                wl_results = sorted(
                    [r for r in dtype_results if r.workload_name == wl_name],
                    key=lambda r: r.batch_size,
                )

                if not wl_results:
                    continue

                batch_sizes = [r.batch_size for r in wl_results]
                tflops_vals = [r.tflops for r in wl_results]
                theory_vals = [r.theory_tflops for r in wl_results]
                mfu_vals = [r.mfu * 100 for r in wl_results]

                cat = wl_results[0].category
                color = category_colors.get(cat, '#666666')
                marker = category_markers.get(cat, 'o')

                # Plot actual TFLOPS
                ax.plot(batch_sizes, tflops_vals,
                        marker=marker, color=color, linewidth=2,
                        markersize=6, label='Actual TFLOPS', zorder=3)

                # Plot theory TFLOPS (roofline)
                ax.plot(batch_sizes, theory_vals,
                        marker='', color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6, label='Roofline TFLOPS', zorder=2)

                # Plot hardware peak line
                if hw_tflops > 0:
                    ax.axhline(y=hw_tflops, color='red', linewidth=1,
                               linestyle=':', alpha=0.5, label=f'HW Peak ({hw_tflops:.0f})')

                # Annotate MFU on data points
                for i, (bs, tf, mfu) in enumerate(zip(batch_sizes, tflops_vals, mfu_vals)):
                    if i % max(1, len(batch_sizes) // 6) == 0 or i == len(batch_sizes) - 1:
                        ax.annotate(
                            f'{mfu:.0f}%',
                            (bs, tf),
                            textcoords="offset points",
                            xytext=(0, 10),
                            fontsize=7,
                            ha='center',
                            color='gray',
                        )

                ax.set_xscale('log', base=2)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xlabel('Batch Size (tokens)', fontsize=10)
                ax.set_ylabel('TFLOPS', fontsize=10)

                # Build title with matrix shape info
                desc = wl_results[0].description
                sample = wl_results[0]
                shape_info = f"N={sample.n}, K={sample.k}"
                ax.set_title(f'{wl_name}\n({shape_info})', fontsize=11, fontweight='bold')

                ax.legend(fontsize=8, loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

            # Hide empty subplots
            for idx in range(n_workloads, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(
                f'LLM GEMM Benchmark: {model_name} | {dtype_str} | {self.device_name}',
                fontsize=14, fontweight='bold', y=1.02,
            )
            plt.tight_layout()

            # Adjust output path for multi-dtype
            if len(dtypes) > 1:
                base, ext = output_path.rsplit('.', 1)
                save_path = f"{base}_{dtype_str}.{ext}"
            else:
                save_path = output_path

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Batch-TFLOPS plot saved to: {save_path}")

    def save_csv(self, results: List[LLMGemmResult], path: str):
        """Save LLM GEMM benchmark results to CSV file."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'model', 'workload', 'category', 'description',
                    'batch_size', 'tp', 'dtype',
                    'M', 'N', 'K',
                    'median_time_ms', 'std_time_ms',
                    'tflops', 'theory_tflops', 'mfu_pct',
                    'bandwidth_gbps', 'mbu_pct',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.model_name, r.workload_name,
                        r.category, r.description,
                        r.batch_size, r.tp, r.dtype,
                        r.m, r.n, r.k,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}",
                        f"{r.tflops:.4f}", f"{r.theory_tflops:.4f}",
                        f"{r.mfu*100:.2f}",
                        f"{r.bandwidth_gbps:.4f}", f"{r.mbu*100:.2f}",
                    ])
            print(f"[INFO] LLM GEMM results saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
