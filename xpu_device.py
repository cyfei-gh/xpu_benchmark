"""
xpu_device: 统一封装 NVIDIA CUDA 与 Ascend NPU (torch_npu) 的差异。

所有 benchmark 代码都应通过此模块获取：
    - 设备字符串       : xpu.device_str(local_rank)
    - 设备数量         : xpu.device_count()
    - 设备名           : xpu.get_device_name(idx)
    - 同步             : xpu.synchronize()
    - Event            : xpu.Event(enable_timing=True)
    - set_device       : xpu.set_device(local_rank)
    - distributed 后端 : xpu.dist_backend()   # 'nccl' or 'hccl'
    - 可用性           : xpu.is_available()

这样可以在不侵入业务逻辑的前提下让 GEMM/MemBw/Comm 同时支持 GPU 与 NPU。
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# ---------------------------------------------------------------------
# 后端检测
# ---------------------------------------------------------------------

_BACKEND: str = "cpu"        # 'cuda' | 'npu' | 'cpu'
_TORCH_NPU_IMPORTED: bool = False

# 优先使用 CUDA（NVIDIA 机器上 torch.cuda.is_available 为 True 时才用）
if torch.cuda.is_available():
    _BACKEND = "cuda"
else:
    # 尝试导入 torch_npu（Ascend NPU）
    try:
        import torch_npu  # noqa: F401  # import 后才会挂载 torch.npu
        _TORCH_NPU_IMPORTED = True
        if hasattr(torch, "npu") and torch.npu.is_available():
            _BACKEND = "npu"
    except Exception:
        _TORCH_NPU_IMPORTED = False


def backend() -> str:
    """返回当前后端: 'cuda' | 'npu' | 'cpu'。"""
    return _BACKEND


def is_cuda() -> bool:
    return _BACKEND == "cuda"


def is_npu() -> bool:
    return _BACKEND == "npu"


def is_available() -> bool:
    return _BACKEND in ("cuda", "npu")


# ---------------------------------------------------------------------
# 设备信息
# ---------------------------------------------------------------------

def _mod():
    """返回当前后端对应的 torch 模块 (torch.cuda 或 torch.npu)。"""
    if _BACKEND == "cuda":
        return torch.cuda
    if _BACKEND == "npu":
        return torch.npu
    raise RuntimeError("No XPU (CUDA / NPU) device available.")


def device_count() -> int:
    if not is_available():
        return 0
    return _mod().device_count()


def current_device() -> int:
    return _mod().current_device()


def set_device(idx: int) -> None:
    _mod().set_device(idx)


def synchronize(idx: Optional[int] = None) -> None:
    if idx is None:
        _mod().synchronize()
    else:
        _mod().synchronize(idx)


def empty_cache() -> None:
    try:
        _mod().empty_cache()
    except Exception:
        pass


def get_device_name(idx: int = 0) -> str:
    return _mod().get_device_name(idx)


def get_device_properties(idx: int = 0):
    """返回设备属性，若后端不支持则返回 None。"""
    try:
        return _mod().get_device_properties(idx)
    except Exception:
        return None


def device_str(local_rank: int = 0) -> str:
    """返回 tensor 可用的 device 字符串，例如 'cuda:0' / 'npu:0'。"""
    if _BACKEND == "cuda":
        return f"cuda:{local_rank}"
    if _BACKEND == "npu":
        return f"npu:{local_rank}"
    return "cpu"


def default_device_str() -> str:
    """无 rank 信息时的默认 device 字符串。"""
    return _BACKEND if _BACKEND in ("cuda", "npu") else "cpu"


# ---------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------

def Event(enable_timing: bool = True):
    """统一的 Event 接口。"""
    return _mod().Event(enable_timing=enable_timing)


# ---------------------------------------------------------------------
# distributed backend
# ---------------------------------------------------------------------

def dist_backend() -> str:
    """返回 torch.distributed 对应的通信后端名。"""
    if _BACKEND == "cuda":
        return "nccl"
    if _BACKEND == "npu":
        return "hccl"
    return "gloo"


# ---------------------------------------------------------------------
# 可见设备环境变量（run.sh 会设置 CUDA_VISIBLE_DEVICES / ASCEND_RT_VISIBLE_DEVICES）
# ---------------------------------------------------------------------

def visible_devices_env_name() -> str:
    if _BACKEND == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    if _BACKEND == "npu":
        # Ascend 官方变量名为 ASCEND_RT_VISIBLE_DEVICES
        return "ASCEND_RT_VISIBLE_DEVICES"
    return ""
