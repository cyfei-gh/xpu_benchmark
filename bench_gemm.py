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
    get_peak_tflops, get_peak_bandwidth, get_device_prefix
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
# 每个条目: (workload_name, [K, N], split_dim)
#   K          : 输入维度
#   N          : 输出维度
#   split_dim  : 张量并行切分维度
#                1 -> 列并行, 沿 N 切分 (e.g. QKV / Moe_gate_up)
#                0 -> 行并行, 沿 K 切分 (e.g. Proj / Moe_down)
MODEL_SHAPE: Dict[str, List] = {
    "Basic": [
        # ('Mx1024x1024', [1024, 1024], 1),
        ('Mx4096x4096', [4096, 4096], 1),
    ],
    "HY-image-3.0": [
        ('QKV', [4096, 6144], 1),
        ('Proj', [4096, 4096], 0),
        ('Moe_gate_up', [4096, 6144], 1),
        ('Moe_down', [3072, 4096], 0),
    ],
    "HY-3.0": [
        ('QKV', [4096, 6144], 1),
        ('Proj', [4096, 4096], 0),
        ('Moe_gate_up', [4096, 1536], 1),
        ('Moe_down', [768, 4096], 0),
    ],
    "HY-image-3.5": [
        ('QKV', [2048, 5120], 1),
        ('Proj', [5120, 2048], 0),
        ('Moe_gate_up', [2048, 1536], 1),
        ('Moe_down', [768, 2048], 0),
        ('QKV_mot', [3200, 5120], 1),
        ('Proj_mot', [5120, 3200], 0),
        ('Moe_gate_up_mot', [3200, 1536], 1),
        ('Moe_down_mot', [768, 3200], 0),
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
    backend: str = 'torch'    # Backend used: 'torch', 'tilelang', 'triton'
    median_time_ms: float = 0.0
    std_time_ms: float = 0.0
    tflops: float = 0.0
    hw_tflops: float = 0.0
    theory_tflops: float = 0.0     # Theoretical peak TFLOPS for this dtype
    theory_time_ms: float = 0.0    # Theoretical minimum time (ms) = max(compute_time, memory_time)
    mfu: float = 0.0               # Model Flops Utilization (0~1)
    bandwidth_gbps: float = 0.0
    hw_bandwidth: float = 0.0
    mbu: float = 0.0               # Bandwidth utilization (0~1)
    device_name: str = ''
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
        backends: List[str] = None,
    ):
        if not xpu.is_available():
            raise RuntimeError("No XPU (CUDA / NPU) device available.")
        self.device = device if device is not None else xpu.default_device_str()
        self.num_iters = num_iters
        self.dry_run_iters = dry_run_iters
        self.enable_cupti = enable_cupti
        self.device_name = xpu.get_device_name(0)
        
        # Backends configuration
        # Available backends: 'torch', 'theory', 'tilelang', 'triton'
        if backends is None:
            self.backends = ['torch']
        else:
            self.backends = backends
        
        # Validate backends
        valid_backends = ['torch', 'tilelang', 'triton']
        for b in self.backends:
            if b not in valid_backends:
                raise ValueError(f"Invalid backend '{b}'. Valid options: {valid_backends}")
        
        # Direct import backends
        self.tilelang_gemm = None
        self.triton_gemm = None
        
        if 'tilelang' in self.backends:
            from . import kernels_tilelang
            self.tilelang_gemm = kernels_tilelang
        
        if 'triton' in self.backends:
            from . import kernels_triton
            self.triton_gemm = kernels_triton

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
    def _run_torch_gemm(
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

    def _run_torch_backend(
        self,
        m: int, n: int, k: int,
        dtype_str: str, cfg: dict,
        flops: int, total_bytes: int,
        hw_tflops: float, hw_bandwidth: float,
        theory_time_ms: float, theory_tflops: float,
        model_name: str, workload_name: str,
        batch_size: int, tp: int,
    ) -> GemmResult:
        """Run GEMM using PyTorch backend and return GemmResult."""
        a, b, scale_a, scale_b = self._create_gemm_tensors(m, n, k, cfg, self.device)

        def fn():
            return self._run_torch_gemm(a, b, scale_a, scale_b, cfg)

        real_ms, std_ms = bench_gpu_time(
            fn,
            enable_cupti=self.enable_cupti,
            num_iters=self.num_iters,
            dry_run_iters=self.dry_run_iters,
        )

        tflops = (flops / 1e9) / real_ms
        bandwidth_gbps = (total_bytes / 1e6) / real_ms
        mfu = (theory_time_ms / real_ms) if real_ms > 0 else 0.0
        mbu = (bandwidth_gbps / hw_bandwidth) if hw_bandwidth > 0 else 0.0

        return GemmResult(
            m=m, n=n, k=k,
            dtype=dtype_str,
            backend='torch',
            median_time_ms=real_ms,
            std_time_ms=std_ms,
            tflops=tflops,
            hw_tflops=hw_tflops,
            theory_time_ms=theory_time_ms,
            theory_tflops=theory_tflops,
            mfu=mfu,
            bandwidth_gbps=bandwidth_gbps,
            hw_bandwidth=hw_bandwidth,
            mbu=mbu,
            device_name=self.device_name,
            model_name=model_name,
            workload_name=workload_name,
            batch_size=batch_size,
            tp=tp,
        )

    def _run_tilelang_backend(
        self,
        m: int, n: int, k: int,
        dtype_str: str, cfg: dict,
        flops: int, total_bytes: int,
        hw_tflops: float, hw_bandwidth: float,
        theory_time_ms: float, theory_tflops: float,
        model_name: str, workload_name: str,
        batch_size: int, tp: int,
    ) -> GemmResult:
        """Run GEMM using TileLang backend and return GemmResult."""
        in_dtype = cfg["in_dtype"]
        device = self.device
        
        # Create input tensors
        a = torch.randn(m, k, device=device, dtype=torch.float32).to(in_dtype)
        b = torch.randn(k, n, device=device, dtype=torch.float32).to(in_dtype)
        
        # Get TileLang GEMM function
        try:
            tilelang_func = self.tilelang_gemm.create_gemm(
                M=m, N=n, K=k,
                dtype=dtype_str,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create TileLang GEMM: {e}")
        
        def fn():
            return tilelang_func(a, b)
        
        real_ms, std_ms = bench_gpu_time(
            fn,
            enable_cupti=self.enable_cupti,
            num_iters=self.num_iters,
            dry_run_iters=self.dry_run_iters,
        )
        
        tflops = (flops / 1e9) / real_ms
        bandwidth_gbps = (total_bytes / 1e6) / real_ms
        mfu = (theory_time_ms / real_ms) if real_ms > 0 else 0.0
        mbu = (bandwidth_gbps / hw_bandwidth) if hw_bandwidth > 0 else 0.0

        return GemmResult(
            m=m, n=n, k=k,
            dtype=dtype_str,
            backend='tilelang',
            median_time_ms=real_ms,
            std_time_ms=std_ms,
            tflops=tflops,
            hw_tflops=hw_tflops,
            theory_time_ms=theory_time_ms,
            theory_tflops=theory_tflops,
            mfu=mfu,
            bandwidth_gbps=bandwidth_gbps,
            hw_bandwidth=hw_bandwidth,
            mbu=mbu,
            device_name=self.device_name,
            model_name=model_name,
            workload_name=workload_name,
            batch_size=batch_size,
            tp=tp,
        )

    def _run_triton_backend(
        self,
        m: int, n: int, k: int,
        dtype_str: str, cfg: dict,
        flops: int, total_bytes: int,
        hw_tflops: float, hw_bandwidth: float,
        theory_time_ms: float, theory_tflops: float,
        model_name: str, workload_name: str,
        batch_size: int, tp: int,
    ) -> GemmResult:
        """Run GEMM using Triton backend and return GemmResult."""
        in_dtype = cfg["in_dtype"]
        device = self.device
        
        # Create input tensors
        a = torch.randn(m, k, device=device, dtype=torch.float32).to(in_dtype)
        b = torch.randn(k, n, device=device, dtype=torch.float32).to(in_dtype)
        
        # Get Triton GEMM function
        try:
            triton_func = self.triton_gemm.create_gemm(
                M=m, N=n, K=k,
                dtype=dtype_str,
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Triton GEMM: {e}")
        
        def fn():
            return triton_func(a, b)
        
        real_ms, std_ms = bench_gpu_time(
            fn,
            enable_cupti=self.enable_cupti,
            num_iters=self.num_iters,
            dry_run_iters=self.dry_run_iters,
        )
        
        tflops = (flops / 1e9) / real_ms
        bandwidth_gbps = (total_bytes / 1e6) / real_ms
        mfu = (theory_time_ms / real_ms) if real_ms > 0 else 0.0
        mbu = (bandwidth_gbps / hw_bandwidth) if hw_bandwidth > 0 else 0.0

        return GemmResult(
            m=m, n=n, k=k,
            dtype=dtype_str,
            backend='triton',
            median_time_ms=real_ms,
            std_time_ms=std_ms,
            tflops=tflops,
            hw_tflops=hw_tflops,
            theory_time_ms=theory_time_ms,
            theory_tflops=theory_tflops,
            mfu=mfu,
            bandwidth_gbps=bandwidth_gbps,
            hw_bandwidth=hw_bandwidth,
            mbu=mbu,
            device_name=self.device_name,
            model_name=model_name,
            workload_name=workload_name,
            batch_size=batch_size,
            tp=tp,
        )

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
    ) -> List[GemmResult]:
        """
        Run a single GEMM benchmark with multiple backends.

        Args:
            m, n, k: Matrix dimensions. C(m,n) = A(m,k) @ B(k,n)
            dtype_str: Data type string, e.g. fp32, bf16, fp16, int8, fp8_tensorwise, fp8_block_1x32, nvfp4,
            model_name / workload_name / batch_size / tp: 可选的 LLM workload 元信息.

        Returns:
            List of GemmResult (one for each backend in self.backends).
        """
        if dtype_str not in GEMM_DTYPE_MAP:
            print(f"[ERROR] Unsupported dtype: {dtype_str}")
            return []

        cfg = GEMM_DTYPE_MAP[dtype_str]
        results = []
        
        # Compute common metrics
        flops = _compute_gemm_flops(m, n, k)
        total_bytes = _compute_gemm_bytes(m, n, k, cfg)
        hw_tflops = get_peak_tflops(self.device_name, cfg["peak_key"])
        hw_bandwidth = get_peak_bandwidth(self.device_name)
        
        # Theory metrics (for MFU calculation)
        theory_time_ms = max(
            (flops / hw_tflops / 1e9) if hw_tflops > 0 else 0.0,
            (total_bytes / hw_bandwidth / 1e6) if hw_bandwidth > 0 else 0.0,
        )
        theory_tflops = (flops / 1e9) / theory_time_ms if theory_time_ms > 0 else 0.0

        for backend in self.backends:
            try:
                if backend == 'torch':
                    # Torch backend: use PyTorch's native GEMM
                    result = self._run_torch_backend(
                        m, n, k, dtype_str, cfg, flops, total_bytes,
                        hw_tflops, hw_bandwidth, theory_time_ms, theory_tflops,
                        model_name, workload_name, batch_size, tp,
                    )
                    results.append(result)
                    
                elif backend == 'tilelang':
                    # TileLang backend
                    if self.tilelang_gemm is None:
                        print(f"[WARNING] TileLang backend not available, skipping...")
                        continue
                    result = self._run_tilelang_backend(
                        m, n, k, dtype_str, cfg, flops, total_bytes,
                        hw_tflops, hw_bandwidth, theory_time_ms, theory_tflops,
                        model_name, workload_name, batch_size, tp,
                    )
                    results.append(result)
                    
                elif backend == 'triton':
                    # Triton backend
                    if self.triton_gemm is None:
                        print(f"[WARNING] Triton backend not available, skipping...")
                        continue
                    result = self._run_triton_backend(
                        m, n, k, dtype_str, cfg, flops, total_bytes,
                        hw_tflops, hw_bandwidth, theory_time_ms, theory_tflops,
                        model_name, workload_name, batch_size, tp,
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"[ERROR] Backend '{backend}' failed M={m} N={n} K={k} dtype={dtype_str}: {e}")
                continue

        return results

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
        Run LLM GEMM benchmarks across batch sizes with multiple backends.

        直接从 MODEL_SHAPE 中读取 (name, [K, N], split_dim), 按 tp 切分
        TP 切分规则:
            split_dim == 1 (列并行, e.g. QKV / Moe_gate_up): N -> N // tp
            split_dim == 0 (行并行, e.g. Proj   / Moe_down ): K -> K // tp

        Args:
            model_name: Key in MODEL_SHAPE dict (e.g. 'HY-image-3.0', 'DeepSeek-V3').
            batch_sizes: List of batch sizes (= num tokens) to sweep.
                         Defaults to [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].
            dtypes: Data types to test. Defaults to ['bf16'].
            tp: Tensor parallelism degree.

        Returns:
            List of GemmResult (with LLM workload metadata filled, one per backend per config).
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
        print(f"Backends: {self.backends}")
        print(f"Iters: {self.num_iters} (warmup: {self.dry_run_iters})")
        print(f"{'='*120}")

        for dtype_str in dtypes:
            print(f"\n--- dtype: {dtype_str} ---")
            
            # Build header with backend columns - similar to user's reference log format
            # Header: workload, batch, M/N/K, then for each backend show time/TFLOPS/MFU
            total_width = 16 + 6 + 3 + 6*3 + 3  # base columns width
            for backend in self.backends:
                total_width += 3 + 10 + 10 + 8 + 3  # each backend: spaces + time + TFLOPS + MFU + separator
            total_width += 3 + 14  # theory TFLOPS column
            
            header = f"{'workload':<16} {'batch':>6} | {'M':>6} {'N':>6} {'K':>6} |"
            for backend in self.backends:
                header += f"  {backend:<30} |"
            
            # Sub-header: metrics for each backend
            sub_header = f"{'':<16} {'':>6} | {'':>6} {'':>6} {'':>6} |"
            for backend in self.backends:
                sub_header += f"  {'time(ms)':>10} {'TFLOPS':>10} {'MFU':>8} |"
            sub_header += f"  {'':>14}"
            
            # Separator line
            sep_line = "-" * len(header)
            
            print(header)
            print(sub_header)
            print(sep_line)

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

                    single_results = self.run_single(
                        m, n, k, dtype_str,
                        model_name=model_name,
                        workload_name=wl_name,
                        batch_size=batch_size,
                        tp=tp,
                    )

                    # Add all results to the main list
                    results.extend(single_results)

                    # Print comparison in the new format
                    if single_results:
                        # Build a dict of backend -> result for easy lookup
                        result_by_backend = {r.backend: r for r in single_results}
                        
                        line = f"{wl_name:<16} {batch_size:>6} | {m:>6} {n:>6} {k:>6} |"
                        for backend in self.backends:
                            if backend in result_by_backend:
                                r = result_by_backend[backend]
                                line += f"  {r.median_time_ms:>10.3f} {r.tflops:>10.2f} {r.mfu*100:>7.1f}% |"
                            else:
                                line += f"  {'N/A':>10} {'N/A':>10} {'N/A':>8} |"
                        print(line)

        print(f"{'='*120}")
        return results
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_summary(self, results: List[GemmResult]):
        """Print summary grouped by workload type and backend."""
        if not results:
            print("No results to summarize.")
            return

        model_name = results[0].model_name or 'N/A'
        
        # Get unique backends
        backends = sorted(set(r.backend for r in results))
        
        print(f"\n{'='*80}")
        print(f"GEMM Benchmark Summary | Model: {model_name}")
        print(f"Backends: {backends}")
        print(f"{'='*80}")

        # Group by workload name and backend
        workload_names = sorted(set(r.workload_name for r in results))
        
        for wl_name in workload_names:
            print(f"\n  Workload: {wl_name}")
            print(f"  {'-'*76}")
            
            for backend in backends:
                wl_backend_results = [r for r in results 
                                     if r.workload_name == wl_name and r.backend == backend]
                if not wl_backend_results:
                    continue
                    
                best = max(wl_backend_results, key=lambda r: r.tflops)
                worst = min(wl_backend_results, key=lambda r: r.tflops)
                best_shape = f"{best.m}x{best.n}x{best.k}"
                worst_shape = f"{worst.m}x{worst.n}x{worst.k}"
                print(f"    {backend:<12}: "
                      f"best={best_shape} {best.dtype} {best.tflops:.2f} TFLOPS (MFU={best.mfu*100:.1f}%) | "
                      f"worst={worst_shape} {worst.dtype} {worst.tflops:.2f} TFLOPS (MBU={worst.mbu*100:.1f}%)")
            
            # Print backend comparison for this workload
            all_tflops = {}
            for r in [r for r in results if r.workload_name == wl_name]:
                if r.dtype not in all_tflops:
                    all_tflops[r.dtype] = {}
                all_tflops[r.dtype][r.backend] = r.tflops
            
            if len(backends) > 1:
                print(f"    Comparison (TFLOPS):")
                for dtype, backend_tflops in all_tflops.items():
                    tf_str = ", ".join([f"{b}:{backend_tflops.get(b, 0):.2f}" for b in backends])
                    print(f"      {dtype}: {tf_str}")

    def save_csv(self, results: List[GemmResult], path: str):
        """Save GEMM benchmark results to CSV file with backend information."""
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'model', 'workload',
                    'batch_size', 'tp', 'dtype', 'backend',
                    'M', 'N', 'K',
                    'median_time_ms', 'std_time_ms', 'tflops',
                    'theory_tflops', 'mfu_pct',
                    'bandwidth_gbps', 'mbu_pct',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.model_name, r.workload_name,
                        r.batch_size, r.tp, r.dtype, r.backend,
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
        在一张图中绘制 (workload × dtype × backend) 的 batch_size vs TFLOPS 曲线.

        根据变量数量动态选择样式:
        - 只有1种workload，多种dtype、backend: 用不同颜色区分 dtype，用实线/虚线区分 backend
        - 只有1种backend，多种dtype、workload: 用不同颜色区分 dtype，用实线/虚线区分 workload
        - 只有1种dtype，多种backend、workload: 用不同颜色区分 backend，用实线/虚线区分 workload

        整合 legend (workload, dtype, backend, HW Peak), 并绘制 HW Peak 红线.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.ticker import ScalarFormatter
        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping plot.")
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
                    seen.add(x)
                    out.append(x)
            return out

        workloads = _uniq(r.workload_name for r in results)
        dtypes = _uniq(r.dtype for r in results)
        backends = _uniq(r.backend for r in results)

        # Sort backends in consistent order
        backend_order = ['torch', 'tilelang', 'triton']
        backends = sorted(backends, key=lambda x: backend_order.index(x) if x in backend_order else 99)

        # 判断变量数量
        n_workloads = len(workloads)
        n_dtypes = len(dtypes)
        n_backends = len(backends)
        n_variable = sum([n_workloads > 1, n_dtypes > 1, n_backends > 1])

        # ---------- 2. 样式配置 ----------
        # 颜色循环 (用于区分主要变量)
        COLOR_PALETTE = [
            '#2196F3',  # 蓝色
            '#FF9800',  # 橙色
            '#4CAF50',  # 绿色
            '#9C27B0',  # 紫色
            '#F44336',  # 红色
            '#00BCD4',  # 青色
            '#E91E63',  # 粉红
            '#795548',  # 棕色
            '#607D8B',  # 蓝灰
            '#FFEB3B',  # 黄色
        ]

        # 线条样式循环 (用于区分次要变量: 实线、虚线、点划线)
        LINESTYLE_PALETTE = [
            '-',    # 实线
            '--',   # 虚线
            '-.',   # 点划线
            ':',    # 点线
        ]

        # Workload 显示名称映射
        WL_DISPLAY_NAME = {
            'Basic': 'Basic',
            'QKV': 'QKV',
            'Proj': 'Proj',
            'Moe_gate_up': 'MoE Gate/Up',
            'Moe_down': 'MoE Down',
        }

        # Dtype 显示名称映射
        DTYPE_DISPLAY_NAME = {
            'mxfp8': 'MXFP8',
            'nvfp4': 'NVFP4',
            'fp8_tensorwise': 'FP8-TW',
            'fp8_rowwise': 'FP8-RW',
            'bf16': 'BF16',
            'fp16': 'FP16',
            'fp32': 'FP32',
            'int8': 'INT8',
            'mxfp4': 'MXFP4',
        }

        # Marker 样式 (用于区分第三变量或增强可读性)
        MARKER_PALETTE = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']

        # ---------- 3. 根据变量数量决定样式策略 ----------
        # 策略:
        # - 颜色 (color): 区分数量最多的变量，或第一个变化的变量
        # - 线条样式 (linestyle): 区分第二个变化的变量
        # - marker: 可选，增强区分度

        # 确定哪个变量用颜色区分，哪个用线条样式区分
        if n_workloads == 1 and n_dtypes > 1 and n_backends > 1:
            # 场景1: 1种workload，多种dtype、backend
            # 颜色区分 dtype，线条样式区分 backend
            color_by = 'dtype'
            linestyle_by = 'backend'
            color_items = dtypes
            linestyle_items = backends
        elif n_backends == 1 and n_dtypes > 1 and n_workloads > 1:
            # 场景2: 1种backend，多种dtype、workload
            # 颜色区分 workload，线条样式区分 dtype
            color_by = 'workload'
            linestyle_by = 'dtype'
            color_items = workloads
            linestyle_items = dtypes
        elif n_dtypes == 1 and n_backends > 1 and n_workloads > 1:
            # 场景3: 1种dtype，多种backend、workload
            # 颜色区分 workload，线条样式区分 backend
            color_by = 'workload'
            linestyle_by = 'backend'
            color_items = workloads
            linestyle_items = backends
        else:
            # 通用场景: 3种变量都有多种，或只有1种变量变化
            # 优先级: workload > dtype > backend (颜色区分)
            if n_workloads > 1:
                color_by = 'workload'
                color_items = workloads
            elif n_dtypes > 1:
                color_by = 'dtype'
                color_items = dtypes
            else:
                color_by = 'backend'
                color_items = backends

            # 线条样式区分剩下的变量
            if color_by != 'workload' and n_workloads > 1:
                linestyle_by = 'workload'
                linestyle_items = workloads
            elif color_by != 'dtype' and n_dtypes > 1:
                linestyle_by = 'dtype'
                linestyle_items = dtypes
            elif color_by != 'backend' and n_backends > 1:
                linestyle_by = 'backend'
                linestyle_items = backends
            else:
                linestyle_by = None
                linestyle_items = []

        # 构建颜色映射
        def _get_display_name(item, by):
            if by == 'workload':
                return WL_DISPLAY_NAME.get(item, item)
            elif by == 'dtype':
                return DTYPE_DISPLAY_NAME.get(item, item)
            else:
                return item

        color_map = {}  # item -> color
        for i, item in enumerate(color_items):
            color_map[item] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        # 构建线条样式映射
        linestyle_map = {}  # item -> linestyle
        for i, item in enumerate(linestyle_items):
            linestyle_map[item] = LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)]

        # Marker 映射 (可选)
        marker_map = {}
        all_items_for_marker = []
        if n_workloads > 1:
            all_items_for_marker.extend([(w, 'workload') for w in workloads])
        if n_dtypes > 1:
            all_items_for_marker.extend([(d, 'dtype') for d in dtypes])
        if n_backends > 1:
            all_items_for_marker.extend([(b, 'backend') for b in backends])

        for i, (item, _) in enumerate(all_items_for_marker):
            if item not in marker_map:
                marker_map[item] = MARKER_PALETTE[i % len(MARKER_PALETTE)]

        # ---------- 4. HW Peak ----------
        def _peak_for(dtype_str):
            cfg = GEMM_DTYPE_MAP.get(dtype_str, {})
            return get_peak_tflops(self.device_name, cfg.get('peak_key', dtype_str))

        peak_specs = []
        for d in dtypes:
            tf = _peak_for(d)
            if tf > 0:
                peak_specs.append((d, tf))

        PEAK_COLOR = '#D32F2F'  # 红色

        # ---------- 5. 公共绘制 helper ----------
        def _filt(wl, dt, be):
            return sorted(
                (r for r in results if r.workload_name == wl and r.dtype == dt and r.backend == be),
                key=lambda r: r.batch_size,
            )

        def _get_line_style(wl, dt, be):
            """根据当前曲线确定颜色和线条样式."""
            # 确定颜色
            if color_by == 'workload':
                color = color_map.get(wl, '#000000')
            elif color_by == 'dtype':
                color = color_map.get(dt, '#000000')
            else:  # backend
                color = color_map.get(be, '#000000')

            # 确定线条样式
            ls = '-'  # 默认实线
            if linestyle_by == 'workload' and wl in linestyle_map:
                ls = linestyle_map[wl]
            elif linestyle_by == 'dtype' and dt in linestyle_map:
                ls = linestyle_map[dt]
            elif linestyle_by == 'backend' and be in linestyle_map:
                ls = linestyle_map[be]

            # 确定 marker
            marker = 'o'  # 默认圆圈
            if n_variable <= 2:
                # 只有2种变量时，用 marker 增强区分
                if wl in marker_map:
                    marker = marker_map[wl]
                elif dt in marker_map:
                    marker = marker_map[dt]
                elif be in marker_map:
                    marker = marker_map[be]

            return color, ls, marker

        def _get_label(wl, dt, be):
            """生成图例标签."""
            parts = []
            if n_workloads > 1:
                parts.append(_get_display_name(wl, 'workload'))
            if n_dtypes > 1:
                parts.append(_get_display_name(dt, 'dtype'))
            if n_backends > 1:
                parts.append(be)
            return ' - '.join(parts) if parts else be

        def _draw_peaks(ax):
            """绘制 HW Peak 红线，返回 legend handles."""
            handles = []
            for d, tf in peak_specs:
                ax.axhline(y=tf, color=PEAK_COLOR, linewidth=2.0,
                           linestyle='--', alpha=0.8, zorder=2)
                display_name = _get_display_name(d, 'dtype')
                handles.append(Line2D([0], [0], color=PEAK_COLOR, linewidth=2.0,
                                      linestyle='--',
                                      label=f'HW Peak {display_name} ({tf:.0f} TFLOPS)'))
            return handles

        def _finalize(ax):
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel('Batch Size (tokens)', fontsize=12)
            ax.set_ylabel('TFLOPS', fontsize=12)
            # 构建标题：包含 model_name, device_name, dtype种类, backend种类
            dtype_str = ', '.join([_get_display_name(d, 'dtype') for d in dtypes])
            backend_str = ', '.join(backends)
            full_title = f'LLM GEMM Benchmark: {model_name} | {self.device_name}\nDtype: {dtype_str}\nBackend: {backend_str}'
            ax.set_title(full_title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            # 设置 y 轴范围，留出顶部空间给标注
            y_max = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=y_max * 1.15)

        def _save():
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Combined batch-TFLOPS plot saved to: {output_path}")

        # ---------- 6. 绘制曲线 ----------
        fig, ax = plt.subplots(figsize=(14, 8))

        # 用于记录已绘制的曲线 (避免重复 label)
        labels_drawn = set()

        for w in workloads:
            for dt in dtypes:
                for be in backends:
                    rs = _filt(w, dt, be)
                    if not rs:
                        continue
                    xs = [r.batch_size for r in rs]
                    ys = [r.tflops for r in rs]

                    color, ls, marker = _get_line_style(w, dt, be)
                    label = _get_label(w, dt, be)

                    # 避免重复 label
                    if label in labels_drawn:
                        label = None
                    else:
                        labels_drawn.add(label)

                    ax.plot(xs, ys, marker=marker, color=color,
                            linestyle=ls, linewidth=2.0, markersize=6,
                            label=label, zorder=3)

        peak_handles = _draw_peaks(ax)

        # ---------- 7. 整合 Legend ----------
        all_handles = []
        all_labels = []

        # 7.1 按策略添加图例
        # 颜色区分的变量
        if color_by == 'workload' and n_workloads > 1:
            for w in color_items:
                h = Line2D([0], [0], color=color_map[w], linestyle='-',
                           linewidth=2.0, label=_get_display_name(w, 'workload'))
                all_handles.append(h)
                all_labels.append(_get_display_name(w, 'workload'))
        elif color_by == 'dtype' and n_dtypes > 1:
            for d in color_items:
                h = Line2D([0], [0], color=color_map[d], linestyle='-',
                           linewidth=2.0, label=_get_display_name(d, 'dtype'))
                all_handles.append(h)
                all_labels.append(_get_display_name(d, 'dtype'))
        elif color_by == 'backend' and n_backends > 1:
            for be in color_items:
                h = Line2D([0], [0], color=color_map[be], linestyle='-',
                           linewidth=2.0, label=be)
                all_handles.append(h)
                all_labels.append(be)

        # 7.2 线条样式区分的变量
        if linestyle_by == 'workload' and n_workloads > 1:
            for w in linestyle_items:
                h = Line2D([0], [0], color='#333333', linestyle=linestyle_map[w],
                           linewidth=2.0, label=_get_display_name(w, 'workload'))
                all_handles.append(h)
                all_labels.append(_get_display_name(w, 'workload'))
        elif linestyle_by == 'dtype' and n_dtypes > 1:
            for d in linestyle_items:
                h = Line2D([0], [0], color='#333333', linestyle=linestyle_map[d],
                           linewidth=2.0, label=_get_display_name(d, 'dtype'))
                all_handles.append(h)
                all_labels.append(_get_display_name(d, 'dtype'))
        elif linestyle_by == 'backend' and n_backends > 1:
            for be in linestyle_items:
                h = Line2D([0], [0], color='#333333', linestyle=linestyle_map[be],
                           linewidth=2.0, label=be)
                all_handles.append(h)
                all_labels.append(be)

        # 7.3 HW Peak legend
        for h in peak_handles:
            all_handles.append(h)
            all_labels.append(h.get_label())

        # 将合并后的 legend 放在图表左上角
        n_legend_items = len(all_handles)
        ncol = min(n_legend_items, 1)

        ax.legend(
            all_handles, all_labels,
            loc='center left',
            ncol=ncol,
            fontsize=9,
            framealpha=0.9,
        )

        _finalize(ax)
        _save()
