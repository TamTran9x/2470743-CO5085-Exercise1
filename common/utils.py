"""Tiện ích dùng chung: seed, thiết bị, đếm tham số, đo thời gian, đọc/ghi JSON."""
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định mọi nguồn ngẫu nhiên để kết quả có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Tổng số phần tử của các tham số (p.numel())."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def sync(device: torch.device) -> None:
    """Chờ GPU chạy xong trước khi đọc đồng hồ, nếu không thời gian đo sẽ sai."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Dùng với `with Timer(device) as t: ...` rồi đọc t.elapsed (giây)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        sync(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self._start
        return False


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment_info() -> dict:
    """Ghi lại môi trường chạy để đưa vào báo cáo."""
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return info
