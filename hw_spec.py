import torch

# ===================================================================
# Hardware Spec Configuration
# All hardware-specific peak performance data is defined here.
# ===================================================================

# Peak performance specs: bandwidth in GB/s, compute in TFLOPS/TOPS
DEVICE_SPECS = {
    "NVIDIA RTX PRO 5000": {
        "name": "RTX 5000",
        "bandwidth": 1344,
        "memory_capacity": 72,
        "l2_cache": 96,           # 96 MB
        "float32": 65,
        "float16": 250,
        "bfloat16": 250,
        "int8": 500,
        "float8_e4m3fn": 1000,
    },
    "NVIDIA RTX PRO 6000D": {
        "name": "RTX 6000D",
        "bandwidth": 1800,
        "memory_capacity": 96,
        "l2_cache": 96,           # 96 MB
        "float32": 120,
        "float16": 480,
        "bfloat16": 480,
        "int8": 960,
        "float8_e4m3fn": 960,
    },
    "NVIDIA L20": {
        "name": "L20",
        "bandwidth": 864,        # 864 GB/s HBM bandwidth
        "memory_capacity": 48,   # 48 GB
        "l2_cache": 96,          # 96 MB
        "float32": 59.8,         # CUDA core FP32
        "float16": 119.5,        # Tensor Core FP16
        "bfloat16": 119.5,       # Tensor Core BF16
        "int8": 239,             # Tensor Core INT8
        "float8_e4m3fn": 239,    # Tensor Core FP8
    },
    "NVIDIA H20": {
        "name": "H20",
        "bandwidth": 4000,
        "memory_capacity": 96,
        "l2_cache": 60,           # 60 MB
        "float32": 40,
        "float16": 147,
        "bfloat16": 147,
        "int8": 293,
        "float8_e4m3fn": 293,
    },
    "NVIDIA H100 SXM5": {
        "name": "H100 SXM",
        "bandwidth": 3350,
        "memory_capacity": 80,
        "l2_cache": 50,           # 50 MB
        "float32": 67,
        "float16": 989,
        "bfloat16": 989,
        "int8": 1979,
        "float8_e4m3fn": 1979,
    },
    "NVIDIA A100-SXM4-80GB": {
        "name": "A100 SXM",
        "bandwidth": 2000,
        "memory_capacity": 80,
        "l2_cache": 40,           # 40 MB
        "float32": 19.5,
        "float16": 312,
        "bfloat16": 312,
        "int8": 624,
    },
}

# GEMM input -> output dtype mapping per backend
DTYPE_OUTPUT_MAPPING = {
    "nvidia": {
        torch.float32: torch.float32,
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.int8: torch.int32,
        torch.float8_e4m3fn: torch.bfloat16,
    },
}

# String -> torch.dtype mapping
DTYPE_FROM_STR = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}

try:
    DTYPE_FROM_STR["float8_e4m3fn"] = torch.float8_e4m3fn
except AttributeError:
    pass

# Bytes per element for each dtype
DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}
try:
    DTYPE_BYTES[torch.float8_e4m3fn] = 1
except AttributeError:
    pass


def get_device_spec(device_name: str) -> dict:
    """
    Fuzzy match device name to spec entry.
    Returns spec dict or None if not found.
    """
    # Exact match first
    if device_name in DEVICE_SPECS:
        return DEVICE_SPECS[device_name]
    # Partial match
    for key, spec in DEVICE_SPECS.items():
        if key.lower() in device_name.lower() or device_name.lower() in key.lower():
            return spec
    return None


def get_peak_tflops(device_name: str, dtype_str: str) -> float:
    """
    Get theoretical peak TFLOPS for given device and dtype.
    Returns 0.0 if not found.
    """
    spec = get_device_spec(device_name)
    if spec is None:
        return 0.0
    return spec.get(dtype_str, 0.0)


def get_peak_bandwidth(device_name: str) -> float:
    """
    Get theoretical peak memory bandwidth in GB/s.
    Returns 0.0 if not found.
    """
    spec = get_device_spec(device_name)
    if spec is None:
        return 0.0
    return spec.get("bandwidth", 0.0)


def get_l2_cache_size(device_name: str) -> int:
    """
    Get L2 cache size in bytes for the given device.
    Returns 40 MB (conservative default) if device is not found.
    """
    spec = get_device_spec(device_name)
    if spec is None:
        return 40 * 1024 * 1024  # 保守默认值
    l2_mb = spec.get("l2_cache", 40)
    return l2_mb * 1024 * 1024
