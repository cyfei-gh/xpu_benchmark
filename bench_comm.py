#!/usr/bin/env python3
"""
GPU 多卡通信带宽 Benchmark。

使用 torch.distributed 测试多卡间集合通信操作的有效带宽，
支持 1~8 卡，覆盖 AllReduce、AllGather、All2All、All2Allv 四种通信原语。

数据量范围：4KB ~ 512MB，绘制 数据量-带宽 曲线。

Usage (standalone):
    torchrun --nproc_per_node=8 -m xpu_benchmark.bench_comm

Usage (via xpu_benchmark main entry):
    在 config JSON 中添加 "comm" section:
    {
        "comm": {
            "num_iters": 50,
            "dry_run_iters": 10,
            "world_size": [2, 4, 8],
            "operations": ["allreduce", "allgather", "all2all", "all2allv"],
            "dtype": "bfloat16"
        }
    }

    world_size 参数说明：
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

    def __str__(self) -> str:
        size_str = _format_size(self.data_size_bytes)
        return (
            f"op={self.operation:<12} size={size_str:>8} | "
            f"world_size={self.world_size} | dtype={self.dtype:<8} | "
            f"time={self.median_time_ms:.3f}±{self.std_time_ms:.3f}ms | "
            f"bus_bw={self.bus_bandwidth_gbps:.2f}GB/s | "
            f"algo_bw={self.algo_bandwidth_gbps:.2f}GB/s"
        )


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


def _run_all2allv(tensor: torch.Tensor, world_size: int, rank: int, group=None):
    """
    执行 All2Allv (非均匀分块)。

    模拟 MoE 场景中不均匀的 token 分发：
    每个 rank 向不同目标发送不同大小的数据块。
    """
    total_elements = tensor.numel()
    # 生成非均匀的分割方案：线性递增分配
    # rank i 向 rank j 发送 base + j 个 chunk
    base_chunk = total_elements // (world_size + world_size * (world_size - 1) // 2)
    base_chunk = max(1, base_chunk)

    input_splits = []
    for j in range(world_size):
        input_splits.append(base_chunk * (j + 1))

    # 调整最后一个 split 以确保总和等于 total_elements
    current_sum = sum(input_splits)
    if current_sum != total_elements:
        # 重新按比例分配
        ratio = total_elements / current_sum
        input_splits = [max(1, int(s * ratio)) for s in input_splits]
        diff = total_elements - sum(input_splits)
        input_splits[-1] += diff

    # 输出 splits：从其他 rank 接收的数据量
    # 简化处理：假设对称分布（实际 MoE 中可能不对称）
    output_splits = input_splits.copy()

    input_tensor = tensor[:sum(input_splits)]
    output_tensor = torch.empty(sum(output_splits), dtype=tensor.dtype, device=tensor.device)

    dist.all_to_all_single(
        output_tensor, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output_tensor


# ===================================================================
# Benchmark 类
# ===================================================================

class CommBenchmark:
    """
    GPU 多卡通信带宽 Benchmark。

    使用 torch.distributed (NCCL backend) 测试多卡间集合通信操作的有效带宽。
    支持 AllReduce、AllGather、All2All、All2Allv 四种通信原语。

    Example usage:
        bench = CommBenchmark(num_iters=50, dry_run_iters=10)
        results = bench.run(
            sizes_bytes=[4096, 65536, 1048576, 16777216, 536870912],
            operations=['allreduce', 'allgather', 'all2all', 'all2allv'],
            dtype='bfloat16',
        )
        if bench.rank == 0:
            bench.print_summary(results)
            bench.plot(results, 'comm_bw_curve.png')
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
            dist.init_process_group(backend='nccl')

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))

        # 设置当前 GPU
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.device_name = torch.cuda.get_device_name(self.local_rank)

    def _create_sub_group(self, target_world_size: int):
        """
        创建指定大小的 process sub-group。

        选取 rank 0 ~ target_world_size-1 组成子组。
        所有 rank 都必须调用此函数（NCCL 要求），但只有子组内的 rank 会参与通信。

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
        使用 CUDA Events 测量通信操作耗时。

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
        torch.cuda.synchronize()
        dist.barrier(group=group)

        times_ms = []
        for _ in range(num_iters):
            # 所有 rank 同步
            dist.barrier(group=group)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            fn()
            end_event.record()

            torch.cuda.synchronize()
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
        n_elements = data_size_bytes // bpe
        # 确保元素数能被 target_world_size 整除（All2All 需要）
        n_elements = (n_elements // target_world_size) * target_world_size
        n_elements = max(target_world_size, n_elements)
        actual_size_bytes = n_elements * bpe

        try:
            # 分配通信 buffer
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
            if ws < 1:
                if self.rank == 0:
                    print(f"[WARNING] world_size={ws} 无效，跳过")
                continue
            if ws == 1:
                if self.rank == 0:
                    print(f"[WARNING] world_size=1 无通信意义，跳过")
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

    def plot(
        self,
        results: List[CommBwResult],
        output_path: str = 'comm_bw_curve.png',
    ):
        """
        在一张图上绘制所有通信操作的 数据量-带宽 曲线。

        X 轴为数据量 (log scale)，Y 轴为 Bus Bandwidth (GB/s)。
        不同操作用不同颜色和标记区分。

        Args:
            results: 测试结果列表。
            output_path: 输出图片路径。
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

        # 配色方案：为每个 (operation, world_size) 组合分配颜色和样式
        base_colors = {
            'allreduce': '#E53935',
            'allgather': '#1E88E5',
            'all2all':   '#43A047',
            'all2allv':  '#FB8C00',
        }
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
        linestyles = ['-', '--', '-.', ':']

        fig, ax = plt.subplots(figsize=(14, 8))

        # 获取所有 (operation, world_size) 组合
        world_sizes_tested = sorted(set(r.world_size for r in results))
        operations = sorted(set(r.operation for r in results))

        line_idx = 0
        for op in operations:
            for ws_idx, ws in enumerate(world_sizes_tested):
                op_results = sorted(
                    [r for r in results if r.operation == op and r.world_size == ws],
                    key=lambda r: r.data_size_bytes,
                )
                if not op_results:
                    continue

                sizes = [r.data_size_bytes for r in op_results]
                bws = [r.bus_bandwidth_gbps for r in op_results]

                color = base_colors.get(op, '#757575')
                marker = markers[ws_idx % len(markers)]
                linestyle = linestyles[ws_idx % len(linestyles)]

                # 生成标签：如果只有一个 world_size，使用简洁标签
                if len(world_sizes_tested) == 1:
                    label_map = {
                        'allreduce': f'TP{ws}_AllReduce',
                        'allgather': 'AllGather',
                        'all2all': 'All2All',
                        'all2allv': 'All2Allv',
                    }
                    label = label_map.get(op, op)
                else:
                    label_map = {
                        'allreduce': f'TP{ws}_AllReduce',
                        'allgather': f'AllGather(ws={ws})',
                        'all2all': f'All2All(ws={ws})',
                        'all2allv': f'All2Allv(ws={ws})',
                    }
                    label = label_map.get(op, f'{op}(ws={ws})')

                ax.plot(
                    sizes, bws,
                    color=color,
                    marker=marker,
                    markersize=7,
                    linewidth=2.2,
                    linestyle=linestyle,
                    label=label,
                    alpha=0.85,
                )
                line_idx += 1

        # X 轴设置
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Data Size per GPU', fontsize=13)
        ax.set_ylabel('Bus Bandwidth (GB/s)', fontsize=13)

        ws_str = ','.join(str(ws) for ws in world_sizes_tested)
        ax.set_title(
            f'Multi-GPU Communication Bandwidth\n'
            f'{self.device_name} | World Sizes: [{ws_str}] | NCCL Backend',
            fontsize=14, fontweight='bold',
        )

        # 自定义 X 轴刻度标签
        all_sizes = sorted(set(r.data_size_bytes for r in results))
        ax.set_xticks(all_sizes)
        tick_labels = [_format_size(s) for s in all_sizes]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

        ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=10)

        # Y 轴从 0 开始
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 通信带宽曲线图已保存: {output_path}")

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
    """
    独立运行入口，使用 torchrun 启动：
        torchrun --nproc_per_node=8 -m xpu_benchmark.bench_comm
    """
    parser = argparse.ArgumentParser(
        description="GPU Multi-Card Communication Bandwidth Benchmark",
    )
    parser.add_argument(
        '--output', type=str, default='./results/',
        help="输出目录",
    )
    parser.add_argument(
        '--num_iters', type=int, default=50,
        help="测量迭代次数",
    )
    parser.add_argument(
        '--dry_run_iters', type=int, default=10,
        help="预热迭代次数",
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16',
        help="数据类型 (float32, float16, bfloat16)",
    )
    parser.add_argument(
        '--operations', type=str, nargs='+',
        default=['allreduce', 'allgather', 'all2all', 'all2allv'],
        help="要测试的通信操作",
    )
    parser.add_argument(
        '--world_size', type=int, nargs='+',
        default=None,
        help="要测试的 rank 数列表，例如 --world_size 2 4 8。"
             "默认使用 torchrun 启动的全部进程数。",
    )

    args = parser.parse_args()

    bench = CommBenchmark(
        num_iters=args.num_iters,
        dry_run_iters=args.dry_run_iters,
    )

    results = bench.run(
        operations=args.operations,
        dtype=args.dtype,
        world_sizes=args.world_size,
    )

    if bench.rank == 0:
        bench.print_summary(results)

        os.makedirs(args.output, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        csv_path = os.path.join(args.output, f'comm_bw_{timestamp}.csv')
        bench.save_csv(results, csv_path)

        plot_path = os.path.join(args.output, f'comm_bw_{timestamp}.png')
        bench.plot(results, plot_path)

    bench.cleanup()


if __name__ == '__main__':
    main()
