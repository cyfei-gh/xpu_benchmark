#!/usr/bin/env python3
"""
GPU 多卡通信带宽 Benchmark。

使用 torch.distributed 测试多卡间集合通信操作的有效带宽，
支持 1~8 卡，覆盖 AllReduce、AllGather、All2All、All2Allv 四种通信原语。

数据量范围：4KB ~ 512MB，绘制 数据量-带宽 曲线。

Usage (standalone):
    torchrun --nproc_per_node=8 -m xpu_benchmark.bench_comm \\
            --config ./config/basic.json --output ./results/

Usage (via xpu_benchmark main entry):
    在 config JSON 中添加 "comm" section:
    {
        "comm": {
            "num_iters": 50,
            "dry_run_iters": 10,
            "world_sizes": [2, 4, 8],
            "operations": ["allreduce", "allgather", "all2all", "all2allv"],
            "dtype": "bfloat16"
        }
    }

    world_sizes 参数说明：
    - 支持整数或列表，例如 8 或 [1, 2, 4, 8]
    - 列表模式下会依次测试不同 rank 数，通过 NCCL sub-group 实现
    - 每个 world_size 必须 <= torchrun 启动的总进程数
    - world_size=1 时跳过通信测试（单卡无通信意义）
"""

import os
import sys
import time
import csv
import json
import argparse
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from . import xpu_device as xpu
from .hw_spec import get_device_prefix


# ===================================================================
# 数据结构
# ===================================================================

@dataclass
class CommBwResult:
    """单次通信带宽测试结果。"""
    operation: str          # 通信操作名称: allreduce, allgather, all2all, all2allv
    data_size_bytes: int    # 每张卡参与通信的数据量 (bytes)
    world_size: int         # 参与通信的 GPU 数量
    dtype: str              # 数据类型
    median_time_ms: float   # 中位数耗时 (ms)
    std_time_ms: float      # 标准差 (ms)
    bus_bandwidth_gbps: float  # Bus 带宽 (GB/s)
    algo_bandwidth_gbps: float  # 算法带宽 (GB/s)
    device_name: str


# ===================================================================
# 辅助函数
# ===================================================================

def _format_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的字符串。"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def _get_default_sizes() -> List[int]:
    """
    生成默认的数据量列表：4KB ~ 512MB，按 2 的幂次递增。
    """
    sizes = []
    size = 4 * 1024  # 4KB
    max_size = 512 * 1024 * 1024  # 512MB
    while size <= max_size:
        sizes.append(size)
        size *= 2
    return sizes


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    """字符串转 torch.dtype。"""
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype_str}，支持: {list(mapping.keys())}")
    return mapping[dtype_str]


def _bytes_per_element(dtype: torch.dtype) -> int:
    """获取每个元素的字节数。"""
    return {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 4)


def _compute_bus_bandwidth(
    operation: str,
    data_size_bytes: int,
    world_size: int,
    time_s: float,
) -> float:
    """
    计算 Bus Bandwidth (GB/s)。

    Bus bandwidth 考虑了集合通信中实际在总线上传输的数据总量。
    参考 NCCL-tests 的计算方式：
    - AllReduce: 2 * (n-1)/n * S / t  (ring algorithm)
    - AllGather: (n-1)/n * S / t
    - All2All:   (n-1)/n * S / t
    - All2Allv:  (n-1)/n * S / t
    """
    if time_s <= 0 or world_size <= 1:
        return 0.0

    n = world_size
    factor = (n - 1) / n

    if operation == 'allreduce':
        # AllReduce = ReduceScatter + AllGather，总线数据量为 2 * (n-1)/n * S
        total_bytes = 2 * factor * data_size_bytes
    elif operation in ('allgather', 'all2all', 'all2allv'):
        total_bytes = factor * data_size_bytes
    else:
        total_bytes = data_size_bytes

    return (total_bytes / 1e9) / time_s


def _compute_algo_bandwidth(
    operation: str,
    data_size_bytes: int,
    world_size: int,
    time_s: float,
) -> float:
    """
    计算 Algorithm Bandwidth (GB/s)。

    Algorithm bandwidth = 数据量 / 时间，不考虑算法实现细节。
    """
    if time_s <= 0:
        return 0.0
    return (data_size_bytes / 1e9) / time_s


# ===================================================================
# 通信操作封装
# ===================================================================

def _run_allreduce(tensor: torch.Tensor, group=None):
    """执行 AllReduce (sum)。"""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def _run_allgather(tensor: torch.Tensor, world_size: int, group=None):
    """执行 AllGather。"""
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output_list, tensor, group=group)
    return output_list


def _run_all2all(tensor: torch.Tensor, world_size: int, group=None):
    """执行 All2All (均匀分块)。"""
    # 将 tensor 均分为 world_size 份
    chunk_size = tensor.numel() // world_size
    input_list = list(tensor.split(chunk_size))
    output_list = [torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)
                   for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return output_list

def _build_moe_send_matrix(total_elements: int, world_size: int) -> List[List[int]]:
    """
    构建模拟 MoE 场景的 send_matrix。

    send_matrix[i][j] 表示 rank i 向 rank j 发送的元素数量。

    关键特性：
    1. **全局一致性**：基于固定 seed + (i, j) 的确定性伪随机，
       所有 rank 独立计算得到完全相同的 send_matrix，无需通信协商。
    2. **对称匹配**：rank i 的 input_splits[j] == rank j 的 output_splits[i] == send_matrix[i][j]，
       保证 all_to_all_single 不 hang。
    3. **MoE 风格不均匀**：每行使用不均匀权重（模拟 token 被路由到不同专家的负载差异），
       典型热点专家收到 2~4x 平均负载，冷门专家仅收到 0.2~0.5x。
    4. **Row-sum 约束**：每行之和严格等于 total_elements（对应 rank i 发送 buffer 的完整使用）。

    注意：列和（rank j 接收总量）会自然产生不均匀差异，这正是 MoE 场景的特征。

    Args:
        total_elements: 每个 rank 发送 buffer 的总元素数。
        world_size: 参与通信的 rank 数。

    Returns:
        send_matrix: world_size × world_size 的整数矩阵。
    """
    # 使用 numpy 的确定性 RNG，保证所有 rank 独立生成完全一致的矩阵
    rng = np.random.RandomState(seed=0xA11_2A11 ^ (world_size * 131))
    # 为每个 (i, j) 生成权重 ∈ [0.2, 3.0]，呈对数均匀分布以拉大热/冷专家差距
    log_w = rng.uniform(np.log(0.2), np.log(3.0), size=(world_size, world_size))
    weights = np.exp(log_w)

    send_matrix: List[List[int]] = []
    for i in range(world_size):
        w_row = weights[i]
        # 按权重分配元素数，至少每格 1 个（避免空发送导致的驱动实现差异）
        raw = w_row / w_row.sum() * total_elements
        row = [max(1, int(x)) for x in raw]

        # 微调：使 row 之和严格等于 total_elements
        diff = total_elements - sum(row)
        if diff > 0:
            # 把剩余元素加到最大的那一格（通常是热点专家）
            idx = int(np.argmax(row))
            row[idx] += diff
        elif diff < 0:
            # 超出则从最大的几格中减掉（保证仍 >= 1）
            remaining = -diff
            order = np.argsort(row)[::-1]  # 从大到小
            for idx in order:
                take = min(remaining, row[idx] - 1)
                if take > 0:
                    row[idx] -= take
                    remaining -= take
                if remaining == 0:
                    break

        send_matrix.append(row)

    return send_matrix


def _run_all2allv(tensor: torch.Tensor, world_size: int, rank: int, group=None):
    """
    执行 All2Allv (非均匀分块)，模拟 MoE 场景的不均匀 token 分发。

    实现要点（避免 hang）：
      all_to_all_single 要求 rank i 的 input_splits[j] 必须等于 rank j 的
      output_splits[i]，否则对端收发量不匹配会永久等待。

    Args:
        tensor: 发送缓冲区（一维），长度为 total_elements。
        world_size: 参与通信的 rank 数。
        rank: 当前 rank 在子组中的索引（0 ~ world_size-1）。
        group: process group。
    """
    total_elements = tensor.numel()
    send_matrix = _build_moe_send_matrix(total_elements, world_size)
    input_splits = send_matrix[rank] # 发送量
    output_splits = [send_matrix[i][rank] for i in range(world_size)] # 接收量

    total_recv = sum(output_splits)
    output_tensor = torch.empty(total_recv, dtype=tensor.dtype, device=tensor.device)

    dist.all_to_all_single(
        output_tensor, tensor,
        output_split_sizes=output_splits, # 接收量
        input_split_sizes=input_splits, # 发送量
        group=group,
    )
    return output_tensor


# ===================================================================
# Benchmark 类
# ===================================================================

class CommBenchmark:
    """
    GPU/NPU 多卡通信带宽 Benchmark。

    使用 torch.distributed (NCCL / HCCL backend) 测试多卡间集合通信操作的有效带宽。
    支持 AllReduce、AllGather、All2All、All2Allv 四种通信原语。
    """

    SUPPORTED_OPS = ['allreduce', 'allgather', 'all2all', 'all2allv']

    def __init__(
        self,
        num_iters: int = 50,
        dry_run_iters: int = 10,
    ):
        self.num_iters = num_iters
        self.dry_run_iters = dry_run_iters

        # 初始化分布式环境（如果尚未初始化）
        if not dist.is_initialized():
            # 检测是否在 torchrun 环境下（torchrun 会自动设置这些环境变量）
            if 'RANK' not in os.environ:
                # 非 torchrun 环境，自动设置单进程模式的环境变量
                os.environ.setdefault('RANK', '0')
                os.environ.setdefault('WORLD_SIZE', '1')
                os.environ.setdefault('LOCAL_RANK', '0')
                os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
                os.environ.setdefault('MASTER_PORT', '29500')
                print("[WARNING] 未检测到 torchrun 环境，以单卡模式初始化。"
                      "多卡通信测试请使用: torchrun --nproc_per_node=N -m xpu_benchmark.bench_comm")
            # 根据后端自动选择 nccl / hccl
            dist.init_process_group(backend=xpu.dist_backend())

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))

        # 设置当前设备 (CUDA / NPU)
        xpu.set_device(self.local_rank)
        self.device = torch.device(xpu.device_str(self.local_rank))
        self.device_name = xpu.get_device_name(self.local_rank)

    def _create_sub_group(self, target_world_size: int):
        """
        创建指定大小的 process sub-group。

        选取 rank 0 ~ target_world_size-1 组成子组。
        所有 rank 都必须调用此函数（NCCL/HCCL 要求），但只有子组内的 rank 会参与通信。

        Args:
            target_world_size: 子组大小。

        Returns:
            (group, is_member): process group 和当前 rank 是否属于该子组。
        """
        ranks = list(range(target_world_size))
        group = dist.new_group(ranks=ranks)
        is_member = self.rank < target_world_size
        return group, is_member

    def _bench_comm_time(
        self,
        fn,
        group=None,
        num_iters: int = None,
        dry_run_iters: int = None,
    ) -> Tuple[float, float]:
        """
        使用 device Events 测量通信操作耗时.

        所有 rank 同步后开始计时，确保测量的是完整的集合通信时间。

        Args:
            fn: 通信操作函数。
            group: process group（用于 barrier 同步）。

        Returns:
            (median_time_ms, std_time_ms)
        """
        if num_iters is None:
            num_iters = self.num_iters
        if dry_run_iters is None:
            dry_run_iters = self.dry_run_iters

        # Warmup
        for _ in range(dry_run_iters):
            fn()
        xpu.synchronize()
        dist.barrier(group=group)

        times_ms = []
        for _ in range(num_iters):
            # 所有 rank 同步
            dist.barrier(group=group)
            xpu.synchronize()

            start_event = xpu.Event(enable_timing=True)
            end_event = xpu.Event(enable_timing=True)

            start_event.record()
            fn()
            end_event.record()

            xpu.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            times_ms.append(elapsed)

        times_arr = np.array(times_ms)
        return float(np.median(times_arr)), float(np.std(times_arr))

    def run_single(
        self,
        data_size_bytes: int,
        operation: str = 'allreduce',
        dtype_str: str = 'bfloat16',
        target_world_size: int = None,
        group=None,
    ) -> Optional[CommBwResult]:
        """
        运行单次通信带宽测试。

        Args:
            data_size_bytes: 每张卡参与通信的数据量 (bytes)。
            operation: 通信操作名称。
            dtype_str: 数据类型。
            target_world_size: 参与通信的 GPU 数量（使用 sub-group 时）。
            group: process group（使用 sub-group 时）。

        Returns:
            CommBwResult 或 None（失败时）。
        """
        if target_world_size is None:
            target_world_size = self.world_size

        try:
            dtype = _dtype_from_str(dtype_str)
        except ValueError as e:
            if self.rank == 0:
                print(f"[ERROR] {e}")
            return None

        bpe = _bytes_per_element(dtype)
        n_elements = (max(data_size_bytes // bpe // target_world_size, 1)) * target_world_size
        actual_size_bytes = n_elements * bpe

        try:
            # 分配通信 buffer
            if operation == 'allgather':
                n_elements = n_elements // target_world_size # NOTE: allgather 实际操作了 tensor_list[world_size]
            tensor = torch.randn(n_elements, dtype=dtype, device=self.device)

            # 构建通信函数
            if operation == 'allreduce':
                def comm_fn():
                    _run_allreduce(tensor, group=group)
            elif operation == 'allgather':
                def comm_fn():
                    _run_allgather(tensor, target_world_size, group=group)
            elif operation == 'all2all':
                def comm_fn():
                    _run_all2all(tensor, target_world_size, group=group)
            elif operation == 'all2allv':
                def comm_fn():
                    _run_all2allv(tensor, target_world_size, self.rank, group=group)
            else:
                if self.rank == 0:
                    print(f"[ERROR] 不支持的操作: {operation}，"
                          f"支持: {self.SUPPORTED_OPS}")
                return None

            # 测量耗时
            median_ms, std_ms = self._bench_comm_time(comm_fn, group=group)

            # 计算带宽
            time_s = median_ms / 1000.0
            bus_bw = _compute_bus_bandwidth(operation, actual_size_bytes, target_world_size, time_s)
            algo_bw = _compute_algo_bandwidth(operation, actual_size_bytes, target_world_size, time_s)

            return CommBwResult(
                operation=operation,
                data_size_bytes=actual_size_bytes,
                world_size=target_world_size,
                dtype=dtype_str,
                median_time_ms=median_ms,
                std_time_ms=std_ms,
                bus_bandwidth_gbps=bus_bw,
                algo_bandwidth_gbps=algo_bw,
                device_name=self.device_name,
            )

        except Exception as e:
            if self.rank == 0:
                print(f"[ERROR] CommBw failed op={operation} size={_format_size(data_size_bytes)} "
                      f"ws={target_world_size} dtype={dtype_str}: {e}")
            return None

    def run(
        self,
        sizes_bytes: List[int] = None,
        operations: List[str] = None,
        dtype: str = 'bfloat16',
        world_sizes: List[int] = None,
    ) -> List[CommBwResult]:
        """
        运行通信带宽 Benchmark，遍历所有数据量、操作类型和 world_size。

        Args:
            sizes_bytes: 数据量列表 (bytes)。默认 4KB~512MB。
            operations: 通信操作列表。默认全部四种。
            dtype: 数据类型。默认 bfloat16。
            world_sizes: 要测试的 rank 数列表，例如 [1, 2, 4, 8]。
                         默认为 None，使用当前全部 world_size。
                         每个值必须 <= 实际启动的总进程数。

        Returns:
            List[CommBwResult]
        """
        if sizes_bytes is None:
            sizes_bytes = _get_default_sizes()
        if operations is None:
            operations = self.SUPPORTED_OPS.copy()
        if world_sizes is None:
            world_sizes = [self.world_size]

        # 校验 world_sizes
        valid_world_sizes = []
        for ws in world_sizes:
            if ws > self.world_size:
                if self.rank == 0:
                    print(f"[WARNING] world_size={ws} > 实际进程数 {self.world_size}，跳过")
                continue
            if ws <= 1:
                if self.rank == 0:
                    print(f"[WARNING] world_size={ws} 无效，跳过")
                continue
            valid_world_sizes.append(ws)

        if not valid_world_sizes:
            if self.rank == 0:
                print("[ERROR] 没有有效的 world_size 可测试")
            return []

        results = []

        if self.rank == 0:
            print(f"\n{'='*120}")
            print(f"Communication Bandwidth Benchmark | Device: {self.device_name} | "
                  f"Total GPUs: {self.world_size}")
            print(f"Iters: {self.num_iters} (warmup: {self.dry_run_iters}) | "
                  f"dtype: {dtype}")
            print(f"Operations: {operations}")
            print(f"World sizes to test: {valid_world_sizes}")
            print(f"Data sizes: {[_format_size(s) for s in sizes_bytes]}")
            print(f"{'='*120}")

        for target_ws in valid_world_sizes:
            # 创建 sub-group（所有 rank 必须参与调用）
            if target_ws == self.world_size:
                group = None
                is_member = True
            else:
                group, is_member = self._create_sub_group(target_ws)

            if self.rank == 0:
                print(f"\n{'─'*120}")
                print(f"  Testing world_size = {target_ws}")
                print(f"{'─'*120}")
                print(f"{'operation':<12} {'size':>8} | {'world_size':>10} | {'dtype':<8} | "
                      f"{'time(ms)':>14} | {'bus_bw(GB/s)':>12} | {'algo_bw(GB/s)':>13}")
                print(f"{'-'*120}")

            if is_member:
                # 当前 rank 属于子组，参与通信测试
                for operation in operations:
                    for size_bytes in sizes_bytes:
                        result = self.run_single(
                            size_bytes, operation, dtype,
                            target_world_size=target_ws,
                            group=group,
                        )
                        if result is not None:
                            results.append(result)
                            if self.rank == 0:
                                print(
                                    f"{result.operation:<12} "
                                    f"{_format_size(result.data_size_bytes):>8} | "
                                    f"{result.world_size:>10} | "
                                    f"{result.dtype:<8} | "
                                    f"{result.median_time_ms:>8.3f}±"
                                    f"{result.std_time_ms:.3f} | "
                                    f"{result.bus_bandwidth_gbps:>12.2f} | "
                                    f"{result.algo_bandwidth_gbps:>13.2f}"
                                )
                        else:
                            if self.rank == 0:
                                print(f"{operation:<12} {_format_size(size_bytes):>8} | "
                                      f"{'FAILED':>10}")
            else:
                # 当前 rank 不属于子组，等待子组完成
                pass

            # 全局 barrier 确保所有 rank 同步后再进入下一个 world_size 测试
            dist.barrier()

            # 销毁 sub-group（避免资源泄漏）
            if group is not None:
                dist.destroy_process_group(group)

        if self.rank == 0:
            print(f"{'='*120}")

        return results

    def print_summary(self, results: List[CommBwResult]):
        """打印通信带宽测试摘要（仅 rank 0）。"""
        if self.rank != 0:
            return
        if not results:
            print("No results to summarize.")
            return

        print(f"\n{'='*80}")
        print(f"Communication Bandwidth Summary")
        print(f"{'='*80}")

        # 按 (world_size, operation) 分组
        world_sizes_tested = sorted(set(r.world_size for r in results))
        operations = sorted(set(r.operation for r in results))

        for ws in world_sizes_tested:
            print(f"\n  ┌─ World Size = {ws}")
            for op in operations:
                op_results = [r for r in results
                              if r.operation == op and r.world_size == ws]
                if not op_results:
                    continue

                best = max(op_results, key=lambda r: r.bus_bandwidth_gbps)
                print(f"  │  [{op.upper()}]")
                print(f"  │    Peak Bus BW   : {best.bus_bandwidth_gbps:.2f} GB/s "
                      f"(size={_format_size(best.data_size_bytes)})")
                print(f"  │    Peak Algo BW  : {best.algo_bandwidth_gbps:.2f} GB/s")
                print(f"  │    Latency @4KB  : ", end="")
                small_results = [r for r in op_results if r.data_size_bytes <= 4096]
                if small_results:
                    print(f"{small_results[0].median_time_ms:.3f} ms")
                else:
                    print("N/A")
            print(f"  └{'─'*40}")

    def plot_comm_bw(
        self,
        results: List[CommBwResult],
        filename_prefix: str = 'comm_bw',
    ):
        """
        按 world_size 分图绘制 size-BW 曲线。

        - 每个 world_size 保存一张图：`{filename_prefix}_TP{ws}.png`
        - 同图中：颜色区分 operation，线型区分带宽类型（bus=虚线, algo=实线）
        - X 轴: data size per GPU (log scale)
        - Y 轴: bandwidth (GB/s)
        """
        if self.rank != 0:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARNING] matplotlib 未安装，跳过绘图。"
                  "安装命令: pip install matplotlib")
            return

        if not results:
            print("[WARNING] 无结果可绘制。")
            return

        # operation -> 颜色 + 可读 label
        op_style_map = {
            'allreduce': {'color': '#E53935', 'label': 'AllReduce'},
            'allgather': {'color': '#1E88E5', 'label': 'AllGather'},
            'all2all':   {'color': '#43A047', 'label': 'All2All'},
            'all2allv':  {'color': '#FB8C00', 'label': 'All2Allv'},
        }
        default_colors = ['#00BCD4', '#795548', '#607D8B', '#E91E63', '#3F51B5']

        # 带宽类型 -> 线型 + marker
        bw_style_map = {
            'algo': {'linestyle': '-',  'marker': 'o'},
            'bus':  {'linestyle': '-.', 'marker': '^'},
        }

        backend_name = xpu.dist_backend().upper()
        world_sizes_tested = sorted(set(r.world_size for r in results))

        for ws in world_sizes_tested:
            ws_results = [r for r in results if r.world_size == ws]
            if not ws_results:
                continue

            operations = sorted(set(r.operation for r in ws_results))
            fig, ax = plt.subplots(figsize=(13, 7.5))

            for i, op in enumerate(operations):
                op_style = op_style_map.get(op, {
                    'color': default_colors[i % len(default_colors)],
                    'label': op,
                })
                color = op_style['color']
                op_label = op_style['label']

                op_results = sorted(
                    [r for r in ws_results if r.operation == op],
                    key=lambda r: r.data_size_bytes,
                )
                if not op_results:
                    continue

                sizes = [r.data_size_bytes for r in op_results]
                algo_bws = [r.algo_bandwidth_gbps for r in op_results]
                bus_bws = [r.bus_bandwidth_gbps for r in op_results]

                # algo 带宽（实线）
                ax.plot(
                    sizes, algo_bws,
                    color=color,
                    linestyle=bw_style_map['algo']['linestyle'],
                    marker=bw_style_map['algo']['marker'],
                    markersize=6,
                    linewidth=2,
                    label=f'{op_label} [algo]',
                    alpha=0.9,
                )
                # 仅在 algo 曲线上标注带宽数值（GB/s），避免与 bus 曲线文字重叠
                for x, y in zip(sizes, algo_bws):
                    ax.annotate(
                        f'{y:.0f}',
                        xy=(x, y),
                        xytext=(0, 6),
                        textcoords='offset points',
                        fontsize=7,
                        color=color,
                        ha='center',
                        va='bottom',
                        alpha=0.85,
                    )
                # bus 带宽（虚线）
                ax.plot(
                    sizes, bus_bws,
                    color=color,
                    linestyle=bw_style_map['bus']['linestyle'],
                    marker=bw_style_map['bus']['marker'],
                    markersize=6,
                    linewidth=2,
                    label=f'{op_label} [bus]',
                    alpha=0.9,
                )

            ax.set_xscale('log', base=2)
            ax.set_xlabel('Data Size per Rank', fontsize=12)
            ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
            ax.set_title(
                f'{backend_name} Bandwidth | {self.device_name}\n'
                f'world_size = {ws}',
                fontsize=13, fontweight='bold',
            )
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both')
            ax.tick_params(labelsize=10)
            ax.set_ylim(bottom=0)

            all_sizes = sorted(set(r.data_size_bytes for r in ws_results))
            if len(all_sizes) <= 20:
                ax.set_xticks(all_sizes)
                ax.set_xticklabels(
                    [_format_size(s) for s in all_sizes],
                    rotation=45, ha='right', fontsize=8,
                )

            output_path = f'{filename_prefix}_TP{ws}.png'
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Comm BW curve (TP{ws}) saved to: {output_path}")

    def save_csv(self, results: List[CommBwResult], path: str):
        """保存测试结果到 CSV 文件（仅 rank 0）。"""
        if self.rank != 0:
            return

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'device', 'operation', 'data_size_bytes', 'data_size_human',
                    'world_size', 'dtype',
                    'median_time_ms', 'std_time_ms',
                    'bus_bandwidth_gbps', 'algo_bandwidth_gbps',
                ])
                for r in results:
                    writer.writerow([
                        r.device_name, r.operation, r.data_size_bytes,
                        _format_size(r.data_size_bytes),
                        r.world_size, r.dtype,
                        f"{r.median_time_ms:.4f}", f"{r.std_time_ms:.4f}",
                        f"{r.bus_bandwidth_gbps:.4f}", f"{r.algo_bandwidth_gbps:.4f}",
                    ])
            print(f"[INFO] 结果已保存: {path}")
        except Exception as e:
            print(f"[ERROR] 保存 CSV 失败: {e}")

    def cleanup(self):
        """清理分布式环境。"""
        if dist.is_initialized():
            dist.destroy_process_group()


# ===================================================================
# Standalone 入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Multi-Card Communication Bandwidth Benchmark",
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="benchmark 配置文件路径（JSON 格式），需包含 'comm' 段",
    )
    parser.add_argument(
        '--output', type=str, default='./results/',
        help="输出目录",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # 从 config JSON 读取 comm 段参数
    # -----------------------------------------------------------------
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取 config 失败: {args.config} -> {e}")
        sys.exit(1)

    if 'comm' not in config:
        print(f"[WARNING] config 文件中未找到 'comm' 段: {args.config}")
        sys.exit(0)

    cfg = config['comm']
    num_iters = cfg.get('num_iters', 50)
    dry_run_iters = cfg.get('dry_run_iters', 10)
    operations = cfg.get('operations',
                         ['allreduce', 'allgather', 'all2all', 'all2allv'])
    dtype = cfg.get('dtype', 'bfloat16')
    sizes_bytes = cfg.get('sizes_bytes', None)
    world_sizes = cfg.get('world_sizes', None)

    # -----------------------------------------------------------------
    # 执行 benchmark
    # -----------------------------------------------------------------
    bench = CommBenchmark(
        num_iters=num_iters,
        dry_run_iters=dry_run_iters,
    )

    if bench.rank == 0:
        print(f"[INFO] Loaded comm config from: {args.config}")
        print(f"[INFO]   num_iters     = {num_iters}")
        print(f"[INFO]   dry_run_iters = {dry_run_iters}")
        print(f"[INFO]   operations    = {operations}")
        print(f"[INFO]   dtype         = {dtype}")
        print(f"[INFO]   world_sizes   = {world_sizes}")

    results = bench.run(
        sizes_bytes=sizes_bytes,
        operations=operations,
        dtype=dtype,
        world_sizes=world_sizes,
    )

    if bench.rank == 0:
        bench.print_summary(results)

        os.makedirs(args.output, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        device_prefix = get_device_prefix(bench.device_name)

        file_name = os.path.join(args.output, f'{device_prefix}_comm_bw_{timestamp}')
        bench.save_csv(results, f'{file_name}.csv')
        bench.plot_comm_bw(results, file_name)

    bench.cleanup()


if __name__ == '__main__':
    main()
