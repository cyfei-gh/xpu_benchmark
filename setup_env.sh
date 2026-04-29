#!/bin/bash

# ------------------------------------------------------------
# 安装 CUDA 版 PyTorch 2.9
# 目标 GPU : RTX PRO 5000 (Blackwell, sm_120)
# 驱动     : 580.105.08  (CUDA 13.0 max)
# 方案     : python venv + pip wheel (cu128)
# ------------------------------------------------------------
# python3 -m pip install torch==2.9.0 --index-url "https://download.pytorch.org/whl/cu128" triton==3.5.0

echo "==> 5) Smoke test"
python3 - <<'PY'
import torch, sys
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("cuda build  :", torch.version.cuda)
if torch.cuda.is_available():
    print("device 0    :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))
print("e4m3 dtype  :", hasattr(torch, "float8_e4m3fn"))
print("e8m0 dtype  :", hasattr(torch, "float8_e8m0fnu"))
print("fp4x2 dtype :", hasattr(torch, "float4_e2m1fn_x2"))
print("_scaled_mm  :", hasattr(torch, "_scaled_mm"))
PY
