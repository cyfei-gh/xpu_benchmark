#!/usr/bin/env python3
"""
H2D / D2H PCIe Bandwidth Test (PyTorch)

Usage: python3 bench_pcie_bandwidth.py --all --size 1024 --trials 15

"""
import torch
import argparse


def measure_transfer(device_id: int, direction: str, size_mb: int, warmup: int = 3, trials: int = 10):
    """
    direction: 'h2d' (host→device) or 'd2h' (device→host)
    Returns (bandwidth_gbps, avg_ms, times_ms_list)
    """
    import time as _time

    device = torch.device(f"cuda:{device_id}")
    num_bytes = size_mb * 1024 * 1024

    # Uses pinned memory and CUDA events for accurate timing.
    host_tensor = torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    dev_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=device)

    # ---- warmup ----
    for _ in range(warmup):
        if direction == "h2d":
            dev_tensor.copy_(host_tensor)
        else:
            host_tensor.copy_(dev_tensor)
        torch.cuda.synchronize(device)

    # ---- measured trials (perf_counter + sync) ----
    times_ms = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        t0 = _time.perf_counter()
        if direction == "h2d":
            dev_tensor.copy_(host_tensor)
        else:
            host_tensor.copy_(dev_tensor)
        torch.cuda.synchronize(device)
        t1 = _time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    bandwidth_gbps = (num_bytes / (avg_ms / 1000)) / 1e9
    return bandwidth_gbps, avg_ms, times_ms


def main():
    parser = argparse.ArgumentParser(description="PCIe H2D/D2H bandwidth test (PyTorch)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Transfer size in MB per copy (default: 1024)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of measured trials (default: 10)")
    parser.add_argument("--all", action="store_true", help="Test all GPUs")
    args = parser.parse_args()

    gpus = list(range(torch.cuda.device_count())) if args.all else [args.gpu]

    # PCIe Gen 5 x16 theoretical unidirectional peak
    # 32 GT/s/lane × 16 lanes × 128b/130b encoding ÷ 8 bits/byte ≈ 63.0 GB/s
    theory_gbps = 63.0

    print(f"\n{'='*90}")
    print(f"  PCIe H2D / D2H Bandwidth Test")
    print(f"  Transfer size: {args.size} MB  |  Trials: {args.trials}  |  GPUs: {gpus}")
    print(f"  Theory peak (PCIe Gen5 x16, unidirectional): {theory_gbps:.1f} GB/s")
    print(f"{'='*90}\n")

    header = f"{'GPU':>4} | {'Dir':>5} | {'Size(MB)':>8} | {'Min(ms)':>8} | {'Avg(ms)':>8} | {'Max(ms)':>8} | {'BW(GB/s)':>10} | {'Efficiency':>10}"
    print(header)
    print("-" * 90)

    for gpu_id in gpus:
        for direction in ["h2d", "d2h"]:
            bw, avg_ms, times = measure_transfer(gpu_id, direction, args.size, trials=args.trials)
            efficiency = (bw / theory_gbps) * 100
            print(f"{gpu_id:>4} | {direction.upper():>5} | {args.size:>8} | "
                  f"{min(times):>8.3f} | {avg_ms:>8.3f} | {max(times):>8.3f} | "
                  f"{bw:>10.2f} | {efficiency:>9.1f}%")
        print("-" * 90)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
