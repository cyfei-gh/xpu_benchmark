#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 CUDA Driver API (Python 绑定) 查询 GPU 的 L2/L1/Shared Memory/Tensor Memory 信息
支持多种 GPU 后端：L20 (Ada, SM 8.9)、H20 (Hopper, SM 9.0)、B200 (Blackwell, SM 10.0) 等

依赖：
    pip install cuda-python          # 推荐，官方 NVIDIA CUDA Python 绑定

用法：
    python3 get_gpu_spec.py               # 查询所有 GPU
    python3 get_gpu_spec.py 0             # 查询指定 device id
"""

from __future__ import annotations

import sys
from typing import Optional


# 先检测 CUDA 可用性；CUDA 不可用时不再强制要求 cuda-python
import os as _os
_HAS_CUDA_PYTHON = False
try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except Exception:
    _CUDA_AVAILABLE = False

# ---------------------------------------------------------------------
# 兼容不同版本的 cuda-python：
#   - cuda-python >= 12.8 : from cuda.bindings import driver
#   - cuda-python <  12.8 : from cuda import cuda as driver
# 仅在 CUDA 可用时才尝试导入，NPU 环境下不依赖 cuda-python。
# ---------------------------------------------------------------------
cu = None  # type: ignore
if _CUDA_AVAILABLE:
    try:
        from cuda.bindings import driver as cu   # 新版 (>=12.8)
        _HAS_CUDA_PYTHON = True
    except ImportError:
        try:
            from cuda import cuda as cu          # 老版
            _HAS_CUDA_PYTHON = True
        except ImportError:
            _HAS_CUDA_PYTHON = False


# ---------------------------------------------------------------------
# CUDA Driver API 调用辅助：Python 绑定返回 (CUresult, *values)，需要解包并检查
# ---------------------------------------------------------------------
def _check(res_tuple):
    """检查 CUDA Driver API 返回值；成功则返回除 CUresult 外的值。"""
    if not isinstance(res_tuple, tuple):
        # 某些函数只返回 CUresult
        res, rest = res_tuple, ()
    else:
        res, *rest = res_tuple

    if int(res) != int(cu.CUresult.CUDA_SUCCESS):
        err_str = cu.cuGetErrorString(res)
        # cuGetErrorString 本身也返回 (CUresult, bytes)
        if isinstance(err_str, tuple):
            err_str = err_str[1]
        if isinstance(err_str, bytes):
            err_str = err_str.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA Driver API error: {err_str} (code={int(res)})")

    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


def get_attr(dev, attr) -> int:
    """查询设备属性；失败返回 -1（与 .cu 版本语义一致）。"""
    try:
        return int(_check(cu.cuDeviceGetAttribute(attr, dev)))
    except Exception:
        return -1


# ---------------------------------------------------------------------
# 架构信息表
# ---------------------------------------------------------------------
def arch_name(major: int, minor: int) -> str:
    """根据 compute capability 返回架构名称。"""
    if major == 7:
        return {
            0: "Volta (V100)",
            2: "Volta (Xavier)",
            5: "Turing (T4/RTX20)",
        }.get(minor, "Volta/Turing")
    if major == 8:
        return {
            0: "Ampere (A100)",
            6: "Ampere (A10/A30/A40/RTX30)",
            7: "Ampere (Orin)",
            9: "Ada Lovelace (L20/L40/RTX40)",
        }.get(minor, "Ampere/Ada")
    if major == 9:
        return "Hopper (H100/H20/H200)" if minor == 0 else "Hopper"
    if major == 10:
        return {
            0: "Blackwell Datacenter (B200/B100)",
            1: "Blackwell (GB10)",
        }.get(minor, "Blackwell")
    if major == 12:
        return "Blackwell (RTX50)"
    return "Unknown"


def tensor_memory_per_sm_bytes(major: int, minor: int) -> int:
    """Blackwell (SM 10.0+) 每 SM 固定 256 KB TMEM；早期架构无此硬件。"""
    if major >= 10:
        return 256 * 1024
    return 0


def unified_l1_smem_per_sm_bytes(major: int, minor: int) -> int:
    """
    每 SM 上 L1 与 Shared Memory 合并的 Unified On-chip SRAM 总大小（硬件规格）。
    数据来源：NVIDIA 官方架构白皮书。
    """
    if major == 7 and minor == 0:
        return 128 * 1024            # V100
    if major == 7 and minor == 5:
        return 96 * 1024             # Turing
    if major == 8 and minor == 0:
        return 192 * 1024            # A100
    if major == 8 and minor in (6, 7):
        return 128 * 1024            # GA10x / Orin
    if major == 8 and minor == 9:
        return 128 * 1024            # Ada (L20/L40/RTX40)
    if major == 9 and minor == 0:
        return 256 * 1024            # Hopper (H100/H20)
    if major >= 10:
        return 256 * 1024            # Blackwell (B200 等, 估计值)
    return 0


# ---------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------
def print_size(label: str, nbytes: Optional[int]) -> None:
    if nbytes is None or nbytes <= 0:
        print(f"  {label:<45s} : N/A")
        return
    if nbytes >= (1 << 20):
        print(f"  {label:<45s} : {nbytes:>10d} B  ({nbytes / 1024 / 1024:.2f} MB)")
    elif nbytes >= (1 << 10):
        print(f"  {label:<45s} : {nbytes:>10d} B  ({nbytes / 1024:.2f} KB)")
    else:
        print(f"  {label:<45s} : {nbytes:>10d} B")


# ---------------------------------------------------------------------
# 设备查询
# ---------------------------------------------------------------------
def print_gpu_spec(dev_id: int) -> None:
    dev = _check(cu.cuDeviceGet(dev_id))

    # cuDeviceGetName(name_buffer_len, device) -> (CUresult, bytes)
    name_bytes = _check(cu.cuDeviceGetName(256, dev))
    if isinstance(name_bytes, bytes):
        name = name_bytes.decode("utf-8", errors="replace").rstrip("\x00").strip()
    else:
        name = str(name_bytes)

    A = cu.CUdevice_attribute
    cc_major = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    cc_minor = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    sm_count = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    total_mem = int(_check(cu.cuDeviceTotalMem(dev)))

    print("=" * 71)
    print(f"Device {dev_id}: {name}")
    print(f"  Compute Capability : {cc_major}.{cc_minor}  "
          f"[{arch_name(cc_major, cc_minor)}]")
    print(f"  SM Count           : {sm_count}")
    print(f"  Global Memory      : {total_mem / (1024 ** 3):.2f} GB")
    print("-" * 71)

    # ============ L2 Cache ============
    l2_bytes       = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    l2_max_persist = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)
    # MAX_ACCESS_POLICY_WINDOW_SIZE 在老 bindings 里可能叫 MAX_ACCESS_POLICY_WINDOW_SIZE
    l2_max_access  = -1
    for attr_name in (
        "CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE",
    ):
        if hasattr(A, attr_name):
            l2_max_access = get_attr(dev, getattr(A, attr_name))
            break

    print("[L2 Cache]")
    print_size("L2 Cache Size (device-wide)",          l2_bytes)
    print_size("L2 Max Persisting Cache Size",         l2_max_persist)
    print_size("L2 Max Access Policy Window Size",     l2_max_access)

    # ============ Shared Memory ============
    smem_per_block        = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    smem_per_block_optin  = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    smem_per_sm           = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    smem_reserved_per_blk = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)

    print("\n[Shared Memory]")
    print_size("Max Shared Memory per Block (default)", smem_per_block)
    print_size("Max Shared Memory per Block (opt-in)",  smem_per_block_optin)
    print_size("Max Shared Memory per SM",              smem_per_sm)
    print_size("Reserved Shared Memory per Block",      smem_reserved_per_blk)
    total_smem = smem_per_sm * sm_count if smem_per_sm > 0 and sm_count > 0 else 0
    print_size("Total Shared Memory (per SM * SM count)", total_smem)

    # ============ L1 Cache / Unified L1+SMEM ============
    print("\n[L1 Cache / Unified L1+SharedMem]")
    unified_bytes = unified_l1_smem_per_sm_bytes(cc_major, cc_minor)
    print_size("Unified L1+SMEM per SM (HW total, from arch)", unified_bytes)
    l1_approx = max(0, unified_bytes - max(0, smem_per_sm))
    print_size("L1 Cache per SM (approx = Unified - SMEM)", l1_approx)
    total_l1 = l1_approx * sm_count if sm_count > 0 else 0
    print_size("Total L1 Cache (approx, across all SMs)", total_l1)

    global_l1 = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)
    local_l1  = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)
    print(f"  {'Global Loads L1 Cached':<45s} : {'Yes' if global_l1 > 0 else 'No'}")
    print(f"  {'Local Loads L1 Cached':<45s} : {'Yes' if local_l1  > 0 else 'No'}")

    # ============ Tensor Memory (Blackwell+) ============
    print("\n[Tensor Memory (TMEM, Blackwell+)]")
    tmem_per_sm = tensor_memory_per_sm_bytes(cc_major, cc_minor)
    if tmem_per_sm == 0:
        print(f"  {'Tensor Memory per SM':<45s} : "
              f"N/A (not supported on this architecture)")
    else:
        print_size("Tensor Memory per SM", tmem_per_sm)
        print_size("Total Tensor Memory (per SM * SM count)",
                   tmem_per_sm * sm_count)

    # ============ 其它信息 ============
    regs_per_block     = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
    regs_per_sm        = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
    warp_size          = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    max_threads_per_sm = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    mem_bus_width      = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    mem_clock_khz      = get_attr(dev, A.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    print("\n[Other Info]")
    print(f"  {'Warp Size':<45s} : {warp_size}")
    print(f"  {'Max Threads per SM':<45s} : {max_threads_per_sm}")
    print(f"  {'Max Registers per Block':<45s} : {regs_per_block} (4-byte regs)")
    print(f"  {'Max Registers per SM':<45s} : {regs_per_sm} (4-byte regs)")
    print(f"  {'Global Memory Bus Width':<45s} : {mem_bus_width} bits")
    if mem_clock_khz > 0:
        eff_ghz = mem_clock_khz / 1e6 * 2.0  # DDR -> x2
        bw_gbs  = mem_bus_width / 8.0 * mem_clock_khz * 1e3 * 2.0 / 1e9
        print(f"  {'Memory Clock Rate':<45s} : {eff_ghz:.2f} GHz (effective)")
        print(f"  {'Theoretical Peak HBM/GDDR Bandwidth':<45s} : {bw_gbs:.2f} GB/s")

    print("=" * 71)
    print()


def _try_print_npu_spec(dev_id: Optional[int] = None) -> bool:
    """
    尝试通过 torch_npu 打印 NPU 设备的简要规格。
    成功返回 True, 否则返回 False。
    """
    try:
        import torch
        import torch_npu  # noqa: F401  # 导入后 torch.npu 才挂载
    except Exception as e:
        sys.stderr.write(f"[INFO] torch_npu 不可用: {e}\n")
        return False

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        sys.stderr.write("[INFO] 未检测到可用的 NPU 设备。\n")
        return False

    dev_count = torch.npu.device_count()
    try:
        npu_ver = torch_npu.__version__  # type: ignore
    except Exception:
        npu_ver = "unknown"

    print("=" * 71)
    print(f"Ascend NPU (torch_npu = {npu_ver})")
    print(f"Detected Devices    : {dev_count}")
    print("=" * 71)

    ids = [dev_id] if (dev_id is not None) else list(range(dev_count))
    for i in ids:
        if not (0 <= i < dev_count):
            print(f"[WARNING] invalid npu id: {i}", file=sys.stderr)
            continue
        name = torch.npu.get_device_name(i)
        try:
            props = torch.npu.get_device_properties(i)
            print(f"Device {i}: {name}")
            print(f"props: {props}")
            print("-" * 71)
        except Exception:
            props = None

    return True


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> int:
    # 解析 device id 参数
    dev_id_arg: Optional[int] = None
    if len(sys.argv) >= 2:
        try:
            dev_id_arg = int(sys.argv[1])
        except ValueError:
            print(f"Invalid device id: {sys.argv[1]}", file=sys.stderr)
            return 1

    # 优先走 CUDA 分支（需要 cuda-python 且 CUDA 可用）
    if _CUDA_AVAILABLE and _HAS_CUDA_PYTHON:
        _check(cu.cuInit(0))

        device_count = int(_check(cu.cuDeviceGetCount()))
        if device_count == 0:
            print("No CUDA devices found.")
            return 0

        driver_version = int(_check(cu.cuDriverGetVersion()))
        print(f"CUDA Driver Version : {driver_version // 1000}.{(driver_version % 1000) // 10}")
        print(f"Detected Devices    : {device_count}\n")

        if dev_id_arg is not None:
            if not (0 <= dev_id_arg < device_count):
                print(f"Invalid device id {dev_id_arg} (0..{device_count - 1})", file=sys.stderr)
                return 1
            print_gpu_spec(dev_id_arg)
        else:
            for i in range(device_count):
                print_gpu_spec(i)
        return 0

    # CUDA 不可用或未安装 cuda-python：尝试打印 NPU 信息
    if _try_print_npu_spec(dev_id_arg):
        return 0

    print("[ERROR] 未检测到可用的 CUDA 或 NPU 设备。", file=sys.stderr)
    if not _HAS_CUDA_PYTHON and _CUDA_AVAILABLE:
        print("[HINT] CUDA 可用但未安装 cuda-python，请执行: pip install cuda-python",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
