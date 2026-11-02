"""E2: hai cài đặt Multi-Head Self-Attention (MSA) có cùng giao diện.

    MultiHeadSelfAttentionScratch : tự viết từ nn.Linear, matmul, softmax, reshape.
                                    KHÔNG dùng nn.MultiheadAttention / nn.TransformerEncoder*.
    MultiHeadSelfAttentionTorch   : bọc nn.MultiheadAttention để đối chiếu.

Cả hai nhận x có shape (B, N, D) và trả về (B, N, D).

Công thức (h head, d_k = D / h):
    Q = X W_Q,  K = X W_K,  V = X W_V                       (B, N, D)
    tách head                                               (B, h, N, d_k)
    A = softmax(Q K^T / sqrt(d_k))  theo chiều cuối         (B, h, N, N)
    head_i = A V                                            (B, h, N, d_k)
    ghép head -> (B, N, D),  đầu ra = concat(head) W_O + b_O
"""
import math

import torch
import torch.nn as nn


class MultiHeadSelfAttentionScratch(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phải chia hết cho num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        # Khởi tạo cùng kiểu với nn.MultiheadAttention: xavier_uniform trên ma trận ghép
        # [W_Q; W_K; W_V] kích thước (3D, D), bias = 0; out_proj giữ khởi tạo mặc định của
        # nn.Linear nhưng bias = 0. Nhờ vậy hai cài đặt bắt đầu từ cùng một phân phối trọng số.
        D = self.embed_dim
        w = torch.empty(3 * D, D)
        nn.init.xavier_uniform_(w)
        self.q_proj.weight.copy_(w[:D])
        self.k_proj.weight.copy_(w[D:2 * D])
        self.v_proj.weight.copy_(w[2 * D:])
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if lin.bias is not None:
                lin.bias.zero_()

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = t.shape
        # (B, N, D) -> (B, N, h, d_k) -> (B, h, N, d_k)
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, D = x.shape
        q = self._split_heads(self.q_proj(x))                       # (B, h, N, d_k)
        k = self._split_heads(self.k_proj(x))                       # (B, h, N, d_k)
        v = self._split_heads(self.v_proj(x))                       # (B, h, N, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = torch.softmax(scores, dim=-1)                        # mỗi hàng cộng lại bằng 1
        attn_dropped = self.attn_drop(attn)

        out = torch.matmul(attn_dropped, v)                         # (B, h, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)                  # ghép head: (B, N, D)
        out = self.out_proj(out)
        return (out, attn) if return_attn else out

    @torch.no_grad()
    def load_from_torch(self, mha: nn.MultiheadAttention) -> None:
        """Chép trọng số từ nn.MultiheadAttention để kiểm chứng hai cài đặt cho cùng kết quả.

        mha.in_proj_weight có shape (3D, D) = [W_Q; W_K; W_V] xếp chồng theo hàng.
        """
        D = self.embed_dim
        W = mha.in_proj_weight
        self.q_proj.weight.copy_(W[:D])
        self.k_proj.weight.copy_(W[D:2 * D])
        self.v_proj.weight.copy_(W[2 * D:])
        if mha.in_proj_bias is not None:
            b = mha.in_proj_bias
            self.q_proj.bias.copy_(b[:D])
            self.k_proj.bias.copy_(b[D:2 * D])
            self.v_proj.bias.copy_(b[2 * D:])
        self.out_proj.weight.copy_(mha.out_proj.weight)
        if mha.out_proj.bias is not None:
            self.out_proj.bias.copy_(mha.out_proj.bias)


class MultiHeadSelfAttentionTorch(nn.Module):
    """Bản đối chiếu dùng nn.MultiheadAttention (batch_first=True để nhận (B, N, D))."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout,
                                         bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # need_weights=False cho phép PyTorch dùng kernel scaled_dot_product_attention nhanh hơn.
        out, attn = self.mha(x, x, x, need_weights=return_attn, average_attn_weights=False)
        return (out, attn) if return_attn else out


def build_msa(impl: str, embed_dim: int, num_heads: int, attn_dropout: float = 0.0) -> nn.Module:
    if impl == "scratch":
        return MultiHeadSelfAttentionScratch(embed_dim, num_heads, attn_dropout)
    if impl == "torch":
        return MultiHeadSelfAttentionTorch(embed_dim, num_heads, attn_dropout)
    raise ValueError(f"impl không hợp lệ: {impl}")
