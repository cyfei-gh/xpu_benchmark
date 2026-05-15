"""
TileLang GEMM implementation for benchmarking.
"""

import torch
import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def create_gemm(M: int, N: int, K: int, dtype: str = 'bf16', device: str = 'cuda'):
    """
    Create a TileLang GEMM function for the given shape.
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Data type string ('fp32', 'bf16', 'fp16')
        device: Device to run on (currently only cuda supported)
        
    Returns:
        Callable GEMM function that takes (a, b) and returns output tensor
    """
    # Map dtype string to TileLang dtype and torch dtype
    # 根据 example_gemm.py: dtype=T.float16 但输入是 torch.bfloat16
    if dtype == 'fp32':
        tl_dtype = T.float32
        torch_dtype = torch.float32
    elif dtype == 'bf16':
        tl_dtype = T.bfloat16  # 根据 example_gemm.py
        torch_dtype = torch.bfloat16
    else:  # fp16
        tl_dtype = T.float16
        torch_dtype = torch.float16
    
    accum_dtype = T.float32
    
    # 创建 TileLang kernel
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    kernel = matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, tl_dtype, accum_dtype)  # 使用位置参数
    
    def gemm_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Execute TileLang GEMM: C = A @ B
        """
        # 确保数据类型正确
        if a.dtype != torch_dtype:
            a = a.to(torch_dtype)
        if b.dtype != torch_dtype:
            b = b.to(torch_dtype)
            
        return kernel(a, b)
    
    return gemm_func


class TileLangGemm:
    """TileLang GEMM operator class."""
    
    def __init__(self, M: int, N: int, K: int, dtype: str = 'bf16'):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.gemm_func = create_gemm(M, N, K, dtype)
        
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.gemm_func(a, b)
