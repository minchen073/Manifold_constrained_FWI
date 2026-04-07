"""轻量工具：显存、随机种子、参数量（供 generate / cell 等使用）。"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_memory_usage(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过显存统计。")
        return
    dev = device or torch.device("cuda", 0)
    with torch.cuda.device(dev):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated / (1024 ** 2):.2f} MiB")
    print(f"Reserved memory: {reserved / (1024 ** 2):.2f} MiB")


def calculate_num_params(model: nn.Module) -> str:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params < 1000:
        return str(num_params)
    if num_params < 1000 * 1000:
        return f"{(num_params / 1000):.2f}K"
    if num_params < 1000 * 1000 * 1000:
        return f"{(num_params / (1000 * 1000)):.2f}M"
    return f"{(num_params / (1000 * 1000 * 1000)):.2f}G"


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)
