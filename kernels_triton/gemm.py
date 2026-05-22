"""
Triton GEMM implementation for benchmarking.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Triton GEMM kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_block = min(BLOCK_K, k_remaining)
        
        mask_k = offs_k < k_block
        a_mask = (offs_m[:, None] < M) & (mask_k[None, :])
        b_mask = (offs_k[:, None] < k_block) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton GEMM: C = A @ B
    
    Args:
        a: Input tensor A of shape (M, K)
        b: Input tensor B of shape (K, N)
        
    Returns:
        Output tensor C of shape (M, N)
    """
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Dimension mismatch: A has K={K}, B has K={K_b}"
    
    # Ensure contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch kernel
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    GROUP_M = 8
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return c


def create_gemm(M: int, N: int, K: int, dtype: str = 'bf16', device: str = 'cuda'):
    """
    Create a Triton GEMM function for the given shape.
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Data type string
        device: Device to run on
        
    Returns:
        Callable GEMM function
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    def gemm_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return gemm_triton(a, b)
    
    return gemm_func
