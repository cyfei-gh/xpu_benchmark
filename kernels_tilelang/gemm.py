"""
TileLang GEMM implementation for benchmarking.
"""

import torch
import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[-1])
def gemm_bf16_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    threads,
    enable_swizzle,
    use_c_shared,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """
    Optimized GEMM kernel.

    1. (Optional) C_shared staging: write C_local -> C_shared -> C(global)
       so the store path is fully coalesced 128-byte transactions. We make
       this optional because the staging buffer also costs shared memory,
    2. L2 swizzle (`T.use_swizzle`): improves L2 cache hit rate for the B matrix.
    3. Larger block_N and larger threads (256) in the large-M regime;
       smaller (block_M=32/64) tiles for small/medium M so the resulting
       grid fully populates all 110 SMs on RTX PRO 5000 Blackwell.
    4. block_K=64 with num_stages=2/3: deep enough software pipeline to
       hide global memory latency while fitting the per-block shared-
       memory budget.
    5. Tail-effect mitigation for M not aligned to block_M: smaller
       block_M reduces wasted compute in the last block-row when M is
       only slightly above a multiple of block_M (e.g. M=2050).
    """

    if use_c_shared:

        @T.prim_func
        def gemm_opt1_cshared(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                T.use_swizzle(panel_size=10, enable=enable_swizzle)
                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                # Stage result through shared memory for coalesced global writes.
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return gemm_opt1_cshared

    # Variant without C_shared (saves block_M*block_N*2 bytes of smem).
    @T.prim_func
    def gemm_opt1_direct(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_swizzle)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_opt1_direct


def get_sm120_config(M: int, N: int, K: int, tl_dtype: T.dtype) -> dict:
    """
    Heuristic config selection for gemm_bf16_kernel based on M.

    Hardware target: NVIDIA RTX PRO 5000 Blackwell (sm_120, 110 SMs,
    128KB per-block opt-in shared memory).

    Required smem (bytes), bf16 (2B):
        (block_M*block_K + block_K*block_N) * 2 * num_stages
        + (block_M*block_N) * 2 * use_c_shared

    SM-occupancy heuristic: with N=1024, the grid has
        ceil(M/block_M) * (1024/block_N) blocks.  Aim for >= 110 blocks
        so the first wave fills all SMs, and >= 220 to also amortize
        epilogue/prologue across the second wave.
    """
    # TODO: [256, 1024] has 4*16 = 64 blocks, need splitK,
    # smem = (64*128*2)*2*3 + 64*64*2 = 104 KB
    if M <= 512:
        return dict(
            block_M=64, block_N=64, block_K=128,
            num_stages=3, threads=128, enable_swizzle=True, use_c_shared=True, dtype=tl_dtype
        )

    if M <= 4096 and (M % 128) != 0:
        # small M for a smoother wave distribution,
        # grid at M=2050: 33 * 8 = 264 blocks  (~2.4 waves on 110 SMs)
        # grid at M=4099: 65 * 8 = 520 blocks  (~4.7 waves)
        # smem = 88 KB
        return dict(
            block_M=64, block_N=128, block_K=64,
            num_stages=3, threads=256, enable_swizzle=True, use_c_shared=True, dtype=tl_dtype
        )

    # Large M divisible by 128 (1024, 4096, 5000, 8192) OR M >= 4096:
    #   smem = 96 + 32 = 128KB,
    #   grid at M=1024: 8 * 8 = 64 blocks  (single wave, 64/110 SMs busy)
    #   grid at M=4096: 32 * 8 = 256 blocks  (>= 2 waves)
    #   grid at M=8192: 64 * 8 = 512 blocks  (~5 waves, fully amortized)
    return dict(
        block_M=128, block_N=128, block_K=64,
        num_stages=3, threads=256, enable_swizzle=True, use_c_shared=True, dtype=tl_dtype
    )


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
    if dtype == 'fp32':
        tl_dtype = T.float32
        torch_dtype = torch.float32
    elif dtype == 'bf16':
        tl_dtype = T.bfloat16
        torch_dtype = torch.bfloat16
    # TODO: support FP8, MXFP8, MXFP4,
    elif dtype == 'fp8_tensorwise':
        tl_dtype = T.float8_e4m3fn
        torch_dtype = torch.float8_e4m3fn
    else:
        tl_dtype = T.float16
        torch_dtype = torch.float16

    # 创建 TileLang kernel
    config = get_sm120_config(M, N, K, tl_dtype)
    kernel = gemm_bf16_kernel(M, N, K, **config)
    
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
