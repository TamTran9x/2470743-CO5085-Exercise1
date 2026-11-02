"""Nạp Fashion-MNIST và chia train/val/test cố định, dùng chung cho E1, E2, E3.

- 60.000 ảnh train gốc được hoán vị bằng một seed cố định: 54.000 ảnh train, 6.000 ảnh val.
- 10.000 ảnh test gốc giữ nguyên, chỉ dùng để báo cáo kết quả cuối.
- Mean/std để chuẩn hóa được tính trên phần train (không nhìn val/test).
- Toàn bộ ảnh được giữ sẵn dưới dạng tensor trong bộ nhớ (28x28 nên rất nhẹ),
  nên mỗi epoch không phải đọc lại ảnh từ đĩa.
"""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    mean: float
    std: float
    sizes: dict


def _load_raw(root: str):
    from torchvision import datasets

    train = datasets.FashionMNIST(root, train=True, download=True)
    test = datasets.FashionMNIST(root, train=False, download=True)
    # .data: uint8 (N, 28, 28), .targets: int64 (N,)
    return train.data, train.targets, test.data, test.targets


def split_indices(n_total: int, val_size: int, seed: int):
    """Trả về (train_idx, val_idx) từ một hoán vị cố định."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[val_size:], perm[:val_size]


def get_data(root: str = "data", batch_size: int = 128, val_size: int = 6000,
             split_seed: int = 42, loader_seed: int = 0, eval_batch_size: int = 1000) -> DataBundle:
    x_tr_all, y_tr_all, x_te, y_te = _load_raw(root)
    train_idx, val_idx = split_indices(len(x_tr_all), val_size, split_seed)

    x_tr = x_tr_all[train_idx].float().div_(255.0)
    x_va = x_tr_all[val_idx].float().div_(255.0)
    x_te = x_te.float().div_(255.0)

    mean, std = x_tr.mean().item(), x_tr.std().item()

    def prep(x):  # (N, 28, 28) -> (N, 1, 28, 28), chuẩn hóa theo thống kê của tập train
        return ((x - mean) / std).unsqueeze(1)

    ds_tr = TensorDataset(prep(x_tr), y_tr_all[train_idx])
    ds_va = TensorDataset(prep(x_va), y_tr_all[val_idx])
    ds_te = TensorDataset(prep(x_te), y_te)

    # Generator riêng cho DataLoader: cùng loader_seed -> cùng thứ tự batch giữa các lần chạy.
    g = torch.Generator().manual_seed(loader_seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g, pin_memory=pin)
    val_loader = DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False, pin_memory=pin)

    sizes = {"train": len(ds_tr), "val": len(ds_va), "test": len(ds_te)}
    return DataBundle(train_loader, val_loader, test_loader, mean, std, sizes)


def black_value(mean: float, std: float) -> float:
    """Giá trị của một pixel đen (0) sau khi chuẩn hóa; dùng để che một phần ảnh ở E3."""
    return (0.0 - mean) / std
