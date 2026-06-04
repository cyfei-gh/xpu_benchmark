#!/bin/bash

# ------------------------------------------------------------
# 安装 PyTorch 2.10, cuda 13.0,
# 目标 GPU : RTX PRO 5000 (Blackwell, sm_120)
# 驱动     : 580.105.08  (CUDA 13.0 max)
# ------------------------------------------------------------
# python3 -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0 --index-url "https://download.pytorch.org/whl/cu130"

echo "==> 5) Smoke test"
python3 - <<'PY'
import sys
import os

import torch
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("e4m3 dtype  :", hasattr(torch, "float8_e4m3fn"))
print("e8m0 dtype  :", hasattr(torch, "float8_e8m0fnu"))
print("fp4x2 dtype :", hasattr(torch, "float4_e2m1fn_x2"))
print("_scaled_mm  :", hasattr(torch, "_scaled_mm"))

try:
    import xpu_device as xpu
    print("\n[xpu_device]")
    print("  backend       :", xpu.backend())
    print("  device_count  :", xpu.device_count())
    print("  current_device:", xpu.current_device())

    # 根据设备类型打印不同的设备信息
    if xpu.is_cuda():
        print(f"  device 0    : {torch.cuda.get_device_name(0)}")
        print(f"  arch_capability  : {torch.cuda.get_device_capability(0)}")
        print("  cuda build  :", torch.version.cuda)
    elif xpu.is_npu():
        # import torch_npu
        print(f"  device {0}    : {torch.npu.get_device_name(0)}")
        print(f"  arch_capability  : {torch.npu.get_device_capability(0)}")
        print("  npu build   :", torch.version.cann)
    else:
        print("\n[CPU Mode] No XPU device available")

except ImportError as e:
    print(f"Warning: Failed to import xpu_device: {e}")
PY
