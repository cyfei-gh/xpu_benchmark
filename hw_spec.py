import torch

# ===================================================================
# Hardware Spec Configuration
# All hardware-specific peak performance data is defined here.
# ===================================================================

# Peak performance specs: bandwidth in GB/s, compute in TFLOPS/TOPS
DEVICE_SPECS = {
    "NVIDIA L20": {
        "name": "L20",
        "bandwidth": 864,        # HBM bandwidth, GB/s 
        "memory_capacity": 48,   # HBM capacity, 48 GB
        "l2_cache": 96,          # L2 cache, 96 MB
        "float32": 59.8,         # CUDA core FP32
        "float16": 119.5,        # Tensor Core FP16
        "bfloat16": 119.5,       # Tensor Core BF16
        "int8": 239,             # Tensor Core INT8
        "float8": 239,           # Tensor Core FP8
        "float4": 478,           # Tensor Core FP4
    },
    "NVIDIA RTX PRO 5000": {
        "name": "Pro5000",
        "bandwidth": 1344,
        "memory_capacity": 72,
        "l2_cache": 96,
        "float32": 65,
        "float16": 250,
        "bfloat16": 250,
        "int8": 500,
        "float8": 500,
        "float4": 1000,
    },
    "NVIDIA RTX PRO 6000D": {
        "name": "Pro6000D",
        "bandwidth": 1800,
        "memory_capacity": 96,
        "l2_cache": 96,
        "float32": 120,
        "float16": 480,
        "bfloat16": 480,
        "int8": 960,
        "float8": 960,
        "float4": 1920,
    },
    "NVIDIA H20": {
        "name": "H20",
        "bandwidth": 4000,
        "memory_capacity": 96,
        "l2_cache": 60,
        "float32": 40,
        "float16": 147,
        "bfloat16": 147,
        "int8": 293,
        "float8": 293,
        "float4": 0,
    },
    "NVIDIA H100 SXM5": {
        "name": "H100",
        "bandwidth": 3350,
        "memory_capacity": 80,
        "l2_cache": 50,
        "float32": 67,
        "float16": 989,
        "bfloat16": 989,
        "int8": 1979,
        "float8": 1979,
        "float4": 0,
    },
    "NVIDIA A100-SXM4-80GB": {
        "name": "A100",
        "bandwidth": 2000,
        "memory_capacity": 80,
        "l2_cache": 40,
        "float32": 19.5,
        "float16": 312,
        "bfloat16": 312,
        "int8": 624,
        "float8": 0,
        "float4": 0,
    },
    "Ascend910B": {
        "name": "910B",
        "bandwidth": 1600,
        "memory_capacity": 64,
        "l2_cache": 192,          # 192 MB
        "float32": 67,            # FP32 Cube
        "float16": 320,           # FP16 Cube
        "bfloat16": 320,          # BF16 Cube
        "int8": 640,              # INT8 Cube
        "float8": 0,
        "float4": 0,
    },
    "Ascend950PR": {
        "name": "950PR",
        "bandwidth": 1600,
        "memory_capacity": 128,
        "l2_cache": 112,
        "float32": 50,
        "float16": 400,
        "bfloat16": 400,
        "int8": 800,
        "float8": 800,
        "float4": 1600,
    },
}


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


def get_device_prefix(device_name: str) -> str:
    """
    Get device name prefix from spec.
    Returns empty string if not found.
    """
    spec = get_device_spec(device_name)
    if spec is None:
        return ""
    return spec.get("name", "")


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
