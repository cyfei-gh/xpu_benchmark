"""
GEMM (General Matrix Multiply) benchmark for GPU.

Measures:
- Execution time (ms)
- TFLOPS throughput
- MFU (Model Flops Utilization) relative to theoretical peak
- Memory bandwidth (GB/s)

Supports dtypes: fp32, bf16, fp16, int8, fp8_tensorwise, mxfp8, mxfp4, nvfp4
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .timing import bench_gpu_time
from . import xpu_device as xpu
from .hw_spec import (
    get_peak_tflops, get_peak_bandwidth,
)


# ===================================================================
# Unified GEMM dtype map
#
# 参考 test_fp4_npu.py 中的 QUANT_MATMUL_DATA_TYPE_MAP, 把原来分散的
# DTYPE_FROM_STR / DTYPE_OUTPUT_MAPPING / DTYPE_BYTES 合并成一个总表.
#
# 字段说明:
#   in_dtype     : 输入张量 dtype (A / B)
#   out_dtype    : 输出张量 dtype
#   in_bytes     : 输入每个 *存储元素* 的字节数 (用于带宽计算)
#                  注: float4_e2m1fn_x2 每个存储字节打包了 2 个 FP4, 仍按 1 计;
#                      在带宽计算中按 K/2 维度参与.
#   out_bytes    : 输出每个元素的字节数
#   peak_key     : 在 hw_spec.DEVICE_SPECS 中查找峰值 TFLOPS 用的 key
#   scale_mode   : None / "tensorwise" / "blockwise_1x32" (MX) / "blockwise_1x16" (NVFP4)
#   scale_dtype  : scale 张量 dtype (用于 _scaled_mm), None 表示无 scale
#   scale_block  : scale 沿 K 方向的 block size (32 for MX, 16 for NVFP4)
#   packed_k     : True 表示输入沿 K 维以 2:1 比例打包 (FP4)
# ===================================================================

_HAS_SCALED = hasattr(torch, "_scaled_mm")
FP8_E4M3    = getattr(torch, "float8_e4m3fn",   None)
FP8_E5M2    = getattr(torch, "float8_e5m2",     None)
FP8_E8M0    = getattr(torch, "float8_e8m0fnu",  None)
FP4_E2M1X2  = getattr(torch, "float4_e2m1fn_x2", None)

# torch_npu 依赖 (只在 NPU 后端才会真正加载)
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
    NPU_FP8_E8M0 = getattr(torch_npu, "float8_e8m0fnu", None)
except Exception:
    _HAS_TORCH_NPU = False
    NPU_FP8_E8M0 = None

GEMM_DTYPE_MAP: Dict[str, dict] = {
    "fp32": {
        "in_dtype":  torch.float32,
        "out_dtype": torch.float32,
        "in_bytes": 4, "out_bytes": 4,
        "peak_key": "float32",
        "scale_mode": None, "scale_dtype": None,
        "scale_block": 0, "packed_k": False,
    },
    "bf16": {
        "in_dtype":  torch.bfloat16,
        "out_dtype": torch.bfloat16,
        "in_bytes": 2, "out_bytes": 2,
        "peak_key": "bfloat16",
        "scale_mode": None, "scale_dtype": None,
        "scale_block": 0, "packed_k": False,
    },
    "fp16": {
        "in_dtype":  torch.float16,
        "out_dtype": torch.float16,
        "in_bytes": 2, "out_bytes": 2,
        "peak_key": "float16",
        "scale_mode": None, "scale_dtype": None,
        "scale_block": 0, "packed_k": False,
    },
    "int8": {
        "in_dtype":  torch.int8,
        "out_dtype": torch.int32,
        "in_bytes": 1, "out_bytes": 4,
        "peak_key": "int8",
        "scale_mode": None, "scale_dtype": None,
        "scale_block": 0, "packed_k": False,
    },
    "fp8_tensorwise": {
        "in_dtype":  FP8_E4M3,
        "out_dtype": torch.bfloat16,
        "in_bytes": 1, "out_bytes": 2,
        "peak_key": "float8",
        "scale_mode": "tensorwise", "scale_dtype": torch.float32,
        "scale_block": 0, "packed_k": False,
    },
    "fp8_rowwise": {
        # FP8 row/col-wise: scale_a [M,1] fp32, scale_b [1,N] fp32
        "in_dtype":  FP8_E4M3,
        "out_dtype": torch.bfloat16,
        "in_bytes": 1, "out_bytes": 2,
        "peak_key": "float8",
        "scale_mode": "rowwise", "scale_dtype": torch.float32,
        "scale_block": 0, "packed_k": False,
    },
    "mxfp8": {
        # MXFP8 (OCP): e4m3 data + block-scale (block=32 along K)
        #   GPU: e8m0 scale, shape [round_up(M,128), round_up(K/32, 4)]
        #   NPU: int8 scale, shape [M, K//64, 2]
        "in_dtype":  FP8_E4M3,
        "out_dtype": torch.bfloat16,
        "in_bytes": 1, "out_bytes": 2,
        "peak_key": "float8",
        "scale_mode": "blockwise_1x32", "scale_dtype": FP8_E8M0,
        "scale_block": 32, "packed_k": False,
    },
    "mxfp4": {
        # MXFP4 (OCP): fp4_e2m1fn_x2 (packed 2:1) + block-scale (block=32 along K)
        #   GPU: e8m0 scale (block=32 e8m0), 走 _scaled_mm
        #   NPU: int8 scale, shape [M, K//64, 2], 走 npu_quant_matmul
        "in_dtype":  FP4_E2M1X2,
        "out_dtype": torch.float16,
        "in_bytes": 1, "out_bytes": 2,         # 1 byte 装 2 个 FP4
        "peak_key": "float4",
        "scale_mode": "blockwise_1x32", "scale_dtype": FP8_E8M0,
        "scale_block": 32, "packed_k": True,
    },
    "nvfp4": {
        # NVFP4 (NVIDIA Blackwell): fp4_e2m1fn_x2 + e4m3 block-scale, block=16 along K
        "in_dtype":  FP4_E2M1X2,
        "out_dtype": torch.bfloat16,
        "in_bytes": 1, "out_bytes": 2,
        "peak_key": "float4",
        "scale_mode": "blockwise_1x16", "scale_dtype": FP8_E4M3,
        "scale_block": 16, "packed_k": True,
    },
}


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


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
    cfg: dict,
) -> int:
    """Compute total bytes transferred for a GEMM (read A, read B, write C).

    对于 FP4 (packed_k=True), A/B 的 K 维实际存储为 K/2 个 uint8.
    """
    in_bytes  = cfg["in_bytes"]
    out_bytes = cfg["out_bytes"]
    if cfg.get("packed_k", False):
        bytes_a = m * (k // 2) * in_bytes
        bytes_b = (k // 2) * n * in_bytes
    else:
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
    def _create_gemm_tensors(
        m: int, n: int, k: int,
        cfg: dict,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Create input tensors (and optional scale tensors) for GEMM benchmark.

        约束:
          - A 始终 row-major, shape=(M, K)
          - B 始终 column-major, 通过分配 (N, K) 行主序后 .t() 得到 (K, N) 列主序视图
            (cuBLASLt FP8/FP4 GEMM 的硬性要求, 不要 .contiguous()).
          - 对于 FP4 (packed): 实际存储 shape 为 (M, K//2) / (N, K//2) uint8,
            再 .view(float4_e2m1fn_x2) 得到打包视图.

        scale 构造规则 (统一用 randint 初始化):
          - tensorwise            : (1)             (randint(1,4) -> fp32)
          - rowwise               : (M,1) / (1,N) fp32 (randint(1,4) -> fp32)
          - blockwise_1x32 / 1x16 :
              GPU: [round_up(M,128), round_up(K/blk,4)] e8m0 / e4m3
              NPU: [M, K//64, 2] / [K//64, N, 2] int8, 适配 npu_quant_matmul
        """
        torch.manual_seed(42)
        in_dtype   = cfg["in_dtype"]
        scale_mode = cfg["scale_mode"]
        scale_dt   = cfg["scale_dtype"]
        block      = cfg["scale_block"]

        if in_dtype is None:
            raise RuntimeError(
                f"Required dtype not available in this PyTorch build (cfg={cfg})"
            )

        # ---------- 1. 普通浮点 / int8: 无 scale, 直接返回 ----------
        if in_dtype in (torch.float32, torch.float16, torch.bfloat16):
            a = torch.randn(m, k, device=device, dtype=in_dtype)
            b = torch.randn(k, n, device=device, dtype=in_dtype)
            return a, b, None, None

        if in_dtype == torch.int8:
            a = torch.randint(-128, 127, (m, k), device=device, dtype=torch.int8)
            if xpu.is_npu():
                # NPU npu_quant_matmul 要求 B 列主序 (K, N), 通过 (N, K).t() 得到
                b = torch.randint(-128, 127, (n, k), device=device, dtype=torch.int8).t()
                scale_a = torch.tensor([1.0], device=device, dtype=torch.float32)
                return a, b, scale_a, None
            b = torch.randint(-128, 127, (k, n), device=device, dtype=torch.int8)
            return a, b, None, None

        # ---------- 2. 量化数据张量 (FP8 / FP4) ----------
        if in_dtype == FP8_E4M3:
            a = torch.randn((m, k), device=device, dtype=torch.float32).to(FP8_E4M3)
            b = torch.randn((n, k), device=device, dtype=torch.float32).to(FP8_E4M3).t()
        elif in_dtype == FP4_E2M1X2:
            if k % block != 0 or k % 2 != 0:
                raise ValueError(
                    f"FP4 GEMM requires K divisible by {block} and 2, got K={k}"
                )
            # A: (M, K//2) uint8 -> view fp4_e2m1fn_x2  (row-major)
            a_u8 = torch.randint(0, 256, (m, k // 2), device=device, dtype=torch.uint8)
            a = a_u8.view(FP4_E2M1X2)
            # B: (N, K//2) uint8 -> view fp4_e2m1fn_x2 -> .t() 得到 col-major
            b_u8 = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
            b = b_u8.view(FP4_E2M1X2).t()
        else:
            raise ValueError(f"Unsupported in_dtype: {in_dtype}")

        # ---------- 3. scale 构造 (内联 _build_scales) ----------
        scale_a: Optional[torch.Tensor] = None
        scale_b: Optional[torch.Tensor] = None

        if scale_mode is None:
            return a, b, None, None

        if scale_mode == "tensorwise":
            scale_a = torch.randn((1,), device=device, dtype=scale_dt)
            scale_b = torch.randn((1,), device=device, dtype=scale_dt)
            return a, b, scale_a, scale_b

        if scale_mode == "rowwise":
            if xpu.is_npu():
                scale_a = torch.randn((m,), device=device, dtype=scale_dt)
                scale_b = torch.randn((n,), device=device, dtype=scale_dt)
                return a, b, scale_a, scale_b
            scale_a = torch.randn((m, 1), device=device, dtype=scale_dt)
            scale_b = torch.randn((1, n), device=device, dtype=scale_dt)
            return a, b, scale_a, scale_b

        if scale_mode in ("blockwise_1x32", "blockwise_1x16"):
            # NPU 强制改写为 (M, K//64, 2) / (K//64, N, 2) int8 的 NPU 风格
            if xpu.is_npu():
                if k % 64 != 0:
                    raise ValueError(
                        f"NPU FP8/FP4 quant_matmul requires K divisible by 64, got K={k}"
                    )
                Gp = k // 64
                scale_a = torch.randint(-127, 127, (m, Gp, 2),
                                        device=device, dtype=torch.int8)
                scale_b = torch.randint(-127, 127, (n, Gp, 2),
                                        device=device, dtype=torch.int8).transpose(0, 1)
                return a, b, scale_a, scale_b

            # GPU 路径
            if k % block != 0:
                raise ValueError(f"K={k} must be divisible by {block} for {scale_mode}")
            Mp = _round_up(m, 128)
            Np = _round_up(n, 128)
            Gp = _round_up(k // block, 4)
            scale_a = torch.randint(120, 136, (Mp, Gp),
                                    device=device, dtype=torch.int32).to(scale_dt)
            scale_b = torch.randint(120, 136, (Np, Gp),
                                    device=device, dtype=torch.int32).to(scale_dt)
            return a, b, scale_a, scale_b

        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

    @staticmethod
    def _run_gemm_kernel(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: Optional[torch.Tensor],
        scale_b: Optional[torch.Tensor],
        cfg: dict,
    ) -> torch.Tensor:
        """Execute GEMM for the given dtype config.

        分发规则:
          - bf16/fp16/fp32      : torch.matmul
          - int8                : torch._int_mm (CUDA) / matmul (NPU 回退)
          - FP8 / FP4 + CUDA    : torch._scaled_mm
          - FP8 / FP4 + NPU     : torch_npu.npu_quant_matmul
        """
        in_dtype  = cfg["in_dtype"]
        out_dtype = cfg["out_dtype"]

        # bf16/fp16/fp32: torch.matmul
        if in_dtype in (torch.float32, torch.float16, torch.bfloat16):
            return torch.matmul(a, b)

        # int8 -> int32
        if in_dtype == torch.int8:
            if xpu.is_npu():
                return torch_npu.npu_quant_matmul(a, b, scale_a, output_dtype=torch.float16)
            elif xpu.is_cuda():
                return torch._int_mm(a, b)

        # FP8 / FP4
        if in_dtype not in (FP8_E4M3, FP4_E2M1X2):
            raise ValueError(f"Unsupported dtype for GEMM: {in_dtype}")

        # ---- NPU: torch_npu.npu_quant_matmul (FP8 / MXFP8 / MXFP4 共用) ----
        if xpu.is_npu():
            scale_mode = cfg["scale_mode"]

            # FP8 per-tensor: scale_a = [1] fp32
            if scale_mode == "tensorwise":
                return torch_npu.npu_quant_matmul(
                    a, b, scale_a, output_dtype=out_dtype
                )

            # FP8 per-token + per-channel: scale_a = [M] fp32, scale_b = [N] fp32
            if scale_mode == "rowwise":
                return torch_npu.npu_quant_matmul(
                    a, b, scale_b,
                    pertoken_scale=scale_a,
                    output_dtype=out_dtype,
                )

            # MXFP8 / MXFP4: per-block (block=32 along K) E8M0 scale
            x_dtype = torch.float8_e4m3fn if in_dtype == FP8_E4M3 else torch.float4_e2m1fn_x2
            return torch_npu.npu_quant_matmul(
                a, b, scale_b,
                pertoken_scale=scale_a,
                pertoken_scale_dtype=NPU_FP8_E8M0,
                output_dtype=out_dtype,
                group_sizes=[1, 1, 32], # 固定值
                scale_dtype=NPU_FP8_E8M0,
                x1_dtype=x_dtype,
                x2_dtype=x_dtype,
            )

        # ---- CUDA: torch._scaled_mm ----
        if not _HAS_SCALED:
            raise RuntimeError("torch._scaled_mm not available in this build")
        result = torch._scaled_mm(
            a, b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            use_fast_accum=False,
        )
        # PyTorch 2.9+ 返回 tuple
        if isinstance(result, tuple):
            return result[0]
        return result

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
            dtype_str: Data type string, e.g. fp32, bf16, fp16, int8, fp8_tensorwise, fp8_block_1x32, nvfp4,
            model_name / workload_name / batch_size / tp: 可选的 LLM workload 元信息.

        Returns:
            GemmResult or None on failure.
        """
        if dtype_str not in GEMM_DTYPE_MAP:
            print(f"[ERROR] Unsupported dtype: {dtype_str}")
            return None

        cfg = GEMM_DTYPE_MAP[dtype_str]

        try:
            a, b, scale_a, scale_b = self._create_gemm_tensors(m, n, k, cfg, self.device)

            def fn():
                return self._run_gemm_kernel(a, b, scale_a, scale_b, cfg)

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
        total_bytes = _compute_gemm_bytes(m, n, k, cfg)
        bandwidth_gbps = (total_bytes / 1e6) / real_ms

        # MFU and bandwidth utilization
        hw_tflops = get_peak_tflops(self.device_name, cfg["peak_key"])
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

        直接从 MODEL_SHAPE 中读取 (name, [K, N], split_dim), 按 tp 切分
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
            dtypes = ['bf16']

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
            best_shape = f"{best.m}x{best.n}x{best.k}"
            worst_shape = f"{worst.m}x{worst.n}x{worst.k}"
            print(f"  {wl_name:<14}: "
                  f"best shape={best_shape}, dtype={best.dtype}, {best.tflops:.2f} TFLOPS, MFU={best.mfu*100:.1f}% | "
                  f"worst shape={worst_shape}, dtype={worst.dtype}, {worst.tflops:.2f} TFLOPS, MBU={worst.mbu*100:.1f}%")

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
        在一张图中绘制 (workload × dtype) 的 batch_size vs TFLOPS 曲线.

        三种布局:
          - 单 dtype          : 颜色/marker 区分 workload (单图例)
          - Basic 多 dtype    : 颜色 区分 dtype (单图例, 一个 workload)
          - 多 dtype 多 wl    : 颜色 区分 workload, linestyle 区分 dtype (双图例)

        HW Peak 水平虚线: 优先只画结果中出现的 fp8 / fp4 两类峰值;
                       若都不存在, 退化为为出现过的全部 dtype 各画一条.
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

        # ---------- 1. 元信息 ----------
        model_name = results[0].model_name or 'N/A'

        def _uniq(seq):
            seen, out = set(), []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        workloads = _uniq(r.workload_name for r in results)
        dtypes    = _uniq(r.dtype for r in results)

        # ---------- 2. 样式 ----------
        WL_STYLE_MAP = {
            'Basic':       ('#2196F3', 'o'), 'QKV':       ('#2196F3', 'o'),
            'Proj':        ('#4CAF50', 's'), 'Moe_gate_up': ('#FF9800', 'D'),
            'Moe_down':    ('#9C27B0', '^'), 'QKV_Lora':  ('#2196F3', 'o'),
            'QK_Lora_b':   ('#03A9F4', 'v'), 'V_Lora_b':  ('#009688', '<'),
        }
        FALLBACK = [('#E91E63', 'v'), ('#00BCD4', '<'),
                    ('#795548', '>'), ('#607D8B', 'p')]
        wl_style = {}
        fb = 0
        for w in workloads:
            if w in WL_STYLE_MAP:
                wl_style[w] = WL_STYLE_MAP[w]
            else:
                wl_style[w] = FALLBACK[fb % len(FALLBACK)]; fb += 1

        DTYPE_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0',
                        '#E91E63', '#00BCD4', '#795548', '#607D8B']
        dtype_color = {d: DTYPE_COLORS[i % len(DTYPE_COLORS)]
                       for i, d in enumerate(dtypes)}
        LS_POOL = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        dtype_ls = {d: LS_POOL[i % len(LS_POOL)] for i, d in enumerate(dtypes)}

        # ---------- 3. HW Peak: 优先 fp8 / fp4 ----------
        def _peak_for(dtype_str):
            cfg = GEMM_DTYPE_MAP.get(dtype_str, {})
            return get_peak_tflops(self.device_name, cfg.get('peak_key', dtype_str))

        def _category(dtype_str):
            key = GEMM_DTYPE_MAP.get(dtype_str, {}).get('peak_key', dtype_str)
            if key == 'float8': return 'fp8'
            if key == 'float4': return 'fp4'
            return None

        # 收集 fp8 / fp4 两个分类的代表 dtype (取首个出现的)
        peak_specs = []   # list of (label, tflops)
        for cat in ('fp8', 'fp4'):
            rep = next((d for d in dtypes if _category(d) == cat), None)
            if rep is None:
                continue
            tf = _peak_for(rep)
            if tf > 0:
                peak_specs.append((f'HW Peak {cat.upper()} ({tf:.0f} TFLOPS)', tf))
        if not peak_specs:  # fallback: 给所有 dtype 各画一条
            for d in dtypes:
                tf = _peak_for(d)
                if tf > 0:
                    peak_specs.append((f'HW Peak {d} ({tf:.0f} TFLOPS)', tf))

        PEAK_COLORS = ['#D32F2F', '#388E3C', '#1976D2', '#7B1FA2']

        # ---------- 4. 公共绘制 helper ----------
        def _filt(wl, dt):
            return sorted(
                (r for r in results if r.workload_name == wl and r.dtype == dt),
                key=lambda r: r.batch_size,
            )

        def _annotate_mfu(ax, xs, ys, mfus, color):
            step = max(1, len(xs) // 5)
            for i, (x, y, m) in enumerate(zip(xs, ys, mfus)):
                if i % step == 0 or i == len(xs) - 1:
                    ax.annotate(f'{m:.0f}%', (x, y),
                                textcoords='offset points', xytext=(0, 10),
                                fontsize=7, ha='center', color=color, alpha=0.85)

        def _draw_peaks(ax, linestyle='--', alpha=0.7, lw=1.8):
            handles = []
            for (label, tf), c in zip(peak_specs, PEAK_COLORS):
                ax.axhline(y=tf, color=c, linewidth=lw,
                           linestyle=linestyle, alpha=alpha, zorder=2)
                handles.append(Line2D([0], [0], color=c, linewidth=lw,
                                      linestyle=linestyle, alpha=alpha, label=label))
            return handles

        def _finalize(ax, title):
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel('Batch Size (tokens)', fontsize=12)
            ax.set_ylabel('TFLOPS', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            ax.set_ylim(bottom=0)

        def _save():
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Combined batch-TFLOPS plot saved to: {output_path}")

        # ---------- 5. 三种布局 ----------
        single_dtype = (len(dtypes) == 1)
        basic_mode   = (model_name == 'Basic')

        # ===== A. 单 dtype: 颜色/marker 区分 workload =====
        if single_dtype:
            dt = dtypes[0]
            fig, ax = plt.subplots(figsize=(12, 7))
            for w in workloads:
                rs = _filt(w, dt)
                if not rs: continue
                xs  = [r.batch_size for r in rs]
                ys  = [r.tflops for r in rs]
                mfu = [r.mfu * 100 for r in rs]
                color, marker = wl_style[w]
                ax.plot(xs, ys, marker=marker, color=color,
                        linewidth=2.2, markersize=7, label=w, zorder=3)
                _annotate_mfu(ax, xs, ys, mfu, color)
            _draw_peaks(ax)  # peak 直接画在 ax 上, 沿用 ax.legend 自动收集
            _finalize(ax, f'LLM GEMM Benchmark: {model_name} | {dt} | {self.device_name}')
            # 把 axhline 的 label 一并并入 legend
            for (label, tf), c in zip(peak_specs, PEAK_COLORS):
                ax.plot([], [], color=c, linewidth=1.8, linestyle='--',
                        alpha=0.7, label=label)
            ax.legend(fontsize=10, loc='center left')
            _save()
            return

        # ===== B. Basic 多 dtype: 颜色区分 dtype =====
        if basic_mode:
            wl = workloads[0]
            _, marker = wl_style[wl]
            fig, ax = plt.subplots(figsize=(12, 7))
            for dt in dtypes:
                rs = _filt(wl, dt)
                if not rs: continue
                xs  = [r.batch_size for r in rs]
                ys  = [r.tflops for r in rs]
                mfu = [r.mfu * 100 for r in rs]
                color = dtype_color[dt]
                ax.plot(xs, ys, marker=marker, color=color,
                        linewidth=2.2, markersize=7, label=dt, zorder=3)
                _annotate_mfu(ax, xs, ys, mfu, color)
            for (label, tf), c in zip(peak_specs, PEAK_COLORS):
                ax.axhline(y=tf, color=c, linewidth=1.8, linestyle='--',
                           alpha=0.7, label=label, zorder=2)
            _finalize(ax, f'LLM GEMM Benchmark: {model_name} ({wl}) | {self.device_name}')
            ax.legend(fontsize=9, loc='center left')
            _save()
            return

        # ===== C. 多 wl × 多 dtype: 颜色=wl, linestyle=dtype, 双图例 =====
        fig, ax = plt.subplots(figsize=(13, 7.5))
        for w in workloads:
            color, marker = wl_style[w]
            for dt in dtypes:
                rs = _filt(w, dt)
                if not rs: continue
                xs = [r.batch_size for r in rs]
                ys = [r.tflops for r in rs]
                ax.plot(xs, ys, marker=marker, color=color,
                        linestyle=dtype_ls[dt], linewidth=2.0, markersize=6, zorder=3)
        peak_handles = _draw_peaks(ax, linestyle='--', alpha=0.75, lw=1.8)

        wl_handles = [Line2D([0], [0], color=wl_style[w][0], marker=wl_style[w][1],
                             linestyle='-', linewidth=2.0, markersize=7, label=w)
                      for w in workloads]
        dt_handles = [Line2D([0], [0], color='black', linestyle=dtype_ls[d],
                             linewidth=2.0, label=d) for d in dtypes]

        leg1 = ax.legend(handles=wl_handles, title='Workload',
                         fontsize=9, title_fontsize=10,
                         loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True)
        leg2 = ax.legend(handles=dt_handles, title='Dtype (linestyle)',
                         fontsize=9, title_fontsize=10,
                         loc='upper left', bbox_to_anchor=(1.01, 0.55), frameon=True)
        ax.add_artist(leg1)
        if peak_handles:
            ax.legend(handles=peak_handles, title='HW Peak',
                      fontsize=8, title_fontsize=9,
                      loc='upper left', bbox_to_anchor=(1.01, 0.25), frameon=True)
            ax.add_artist(leg2)

        _finalize(ax, f'LLM GEMM Benchmark: {model_name} | {self.device_name}')
        _save()
