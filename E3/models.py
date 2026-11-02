"""E3: phân loại ảnh bằng LSTM / GRU, coi ảnh là một chuỗi.

Cách biến ảnh (B, 1, 28, 28) thành chuỗi (B, T, F):
    rows    : mỗi hàng là một bước thời gian            -> T = 28, F = 28
    cols    : mỗi cột là một bước thời gian             -> T = 28, F = 28
    patch7  : patch 7x7 duyệt theo hàng (raster order)  -> T = 16, F = 49
    patch4  : patch 4x4 duyệt theo hàng                 -> T = 49, F = 16

Mô hình: chuỗi -> nn.LSTM / nn.GRU (2 tầng, batch_first) -> hidden của bước cuối
-> Linear -> 10 logits.
"""
import torch
import torch.nn as nn


def to_sequence(x: torch.Tensor, mode: str) -> torch.Tensor:
    """(B, C, H, W) -> (B, T, F)."""
    B, C, H, W = x.shape
    if mode == "rows":
        return x.permute(0, 2, 1, 3).reshape(B, H, C * W)
    if mode == "cols":
        return x.permute(0, 3, 1, 2).reshape(B, W, C * H)
    if mode.startswith("patch"):
        p = int(mode[len("patch"):])
        x = x.unfold(2, p, p).unfold(3, p, p)            # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)                   # (B, H/p, W/p, C, p, p)
        return x.reshape(B, (H // p) * (W // p), C * p * p)
    raise ValueError(f"mode không hợp lệ: {mode}")


def sequence_shape(mode: str, C: int = 1, H: int = 28, W: int = 28):
    """(T, F) của từng cách biểu diễn."""
    if mode == "rows":
        return H, C * W
    if mode == "cols":
        return W, C * H
    if mode.startswith("patch"):
        p = int(mode[len("patch"):])
        return (H // p) * (W // p), C * p * p
    raise ValueError(mode)


class RNNClassifier(nn.Module):
    def __init__(self, cell: str = "lstm", seq_mode: str = "rows", hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, num_classes: int = 10):
        super().__init__()
        self.cell, self.seq_mode = cell, seq_mode
        T, F = sequence_shape(seq_mode)
        self.seq_len, self.input_size = T, F
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_size=F, hidden_size=hidden, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):                       # (B, 1, 28, 28)
        seq = to_sequence(x, self.seq_mode)     # (B, T, F)
        out, _ = self.rnn(seq)                  # (B, T, hidden): output tầng trên cùng ở mọi bước
        return self.head(out[:, -1])            # dùng bước cuối -> (B, 10)

    @torch.no_grad()
    def logits_every_step(self, x):
        """Logits nếu dừng ở từng bước t = 1..T: (B, T, 10).

        Vì RNN một chiều, output ở bước t chỉ phụ thuộc t phần đầu của chuỗi, nên đây
        chính là dự đoán khi mô hình mới "nhìn thấy" t hàng/patch đầu tiên.
        """
        out, _ = self.rnn(to_sequence(x, self.seq_mode))
        return self.head(out)


def rnn_param_formula(cell: str, input_size: int, hidden: int, num_layers: int) -> int:
    """Số tham số lý thuyết của nn.LSTM / nn.GRU (PyTorch dùng 2 vector bias mỗi cổng).
    Mỗi cổng ở tầng l: W_ih (H x in_l) + W_hh (H x H) + b_ih (H) + b_hh (H).
    LSTM có 4 cổng (i, f, g, o), GRU có 3 cổng (r, z, n)."""
    gates = {"lstm": 4, "gru": 3}[cell]
    total, in_l = 0, input_size
    for _ in range(num_layers):
        total += gates * (hidden * in_l + hidden * hidden + 2 * hidden)
        in_l = hidden
    return total
