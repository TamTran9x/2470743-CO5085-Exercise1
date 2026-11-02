"""E2: các cách biến ảnh (B, 1, 28, 28) thành chuỗi token (B, N, D).

    patch   : chia lưới ô p x p, mỗi ô là một token (p=4 -> 49 token, p=7 -> 16 token).
              Có thể cho các ô chồng lên nhau bằng stride < p.
    row     : mỗi hàng 28 pixel là một token -> 28 token.
    cnn     : CNN-stem rút feature map 7x7, mỗi vị trí không gian là một token -> 49 token.

Mỗi tokenizer có thuộc tính num_tokens để mô hình tạo positional embedding đúng kích thước.
"""
import torch
import torch.nn as nn

H = W = 28


class PatchTokenizer(nn.Module):
    """Patch embedding.

    nn.Conv2d với kernel_size = stride = p tương đương hoàn toàn với:
    cắt ảnh thành các ô p x p không chồng nhau, flatten mỗi ô (C*p*p giá trị),
    rồi nhân với cùng một ma trận chiếu tuyến tính (C*p*p -> D).
    Viết bằng Conv2d chỉ vì gọn và nhanh hơn.
    """

    def __init__(self, embed_dim: int, patch_size: int = 4, stride: int | None = None, in_ch: int = 1):
        super().__init__()
        stride = stride or patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride)
        side = (H - patch_size) // stride + 1
        self.num_tokens = side * side
        self.grid = (side, side)

    def forward(self, x):                       # (B, C, 28, 28)
        x = self.proj(x)                        # (B, D, side, side)
        return x.flatten(2).transpose(1, 2)     # (B, side*side, D)


class RowTokenizer(nn.Module):
    """Mỗi hàng ảnh (C*28 giá trị) là một token, chiếu tuyến tính lên D chiều."""

    def __init__(self, embed_dim: int, in_ch: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_ch * W, embed_dim)
        self.num_tokens = H

    def forward(self, x):                                   # (B, C, 28, 28)
        B, C, Hh, Ww = x.shape
        rows = x.permute(0, 2, 1, 3).reshape(B, Hh, C * Ww)  # (B, 28, C*28)
        return self.proj(rows)                              # (B, 28, D)


class CNNStemTokenizer(nn.Module):
    """CNN nhỏ: 28x28 -> 14x14 -> 7x7, sau đó conv 1x1 đưa số kênh về D.
    Mỗi vị trí trên lưới 7x7 là một token (49 token), mang thông tin của một vùng ảnh
    lớn hơn một patch 4x4 nhờ receptive field của các tầng conv.
    """

    def __init__(self, embed_dim: int, channels=(32, 64), in_ch: int = 1):
        super().__init__()
        layers, prev = [], in_ch
        for c in channels:
            layers += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        layers.append(nn.Conv2d(prev, embed_dim, kernel_size=1))
        self.stem = nn.Sequential(*layers)
        side = H // (2 ** len(channels))
        self.num_tokens = side * side
        self.grid = (side, side)

    def forward(self, x):
        x = self.stem(x)                        # (B, D, 7, 7)
        return x.flatten(2).transpose(1, 2)     # (B, 49, D)


TOKENIZERS = {
    "patch4": lambda d: PatchTokenizer(d, patch_size=4),
    "patch7": lambda d: PatchTokenizer(d, patch_size=7),
    "patch4s2": lambda d: PatchTokenizer(d, patch_size=4, stride=2),   # patch chồng nhau: 169 token
    "row": lambda d: RowTokenizer(d),
    "cnn": lambda d: CNNStemTokenizer(d),
}


def build_tokenizer(name: str, embed_dim: int) -> nn.Module:
    if name not in TOKENIZERS:
        raise ValueError(f"tokenizer không hợp lệ: {name}; chọn trong {list(TOKENIZERS)}")
    return TOKENIZERS[name](embed_dim)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """Cắt ảnh thành patch bằng thao tác tensor thuần (dùng trong test để chứng minh
    PatchTokenizer == patchify + Linear). (B, C, H, W) -> (B, (H/p)*(W/p), C*p*p)."""
    B, C, Hh, Ww = x.shape
    x = x.unfold(2, p, p).unfold(3, p, p)              # (B, C, H/p, W/p, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5)                     # (B, H/p, W/p, C, p, p)
    return x.reshape(B, (Hh // p) * (Ww // p), C * p * p)
