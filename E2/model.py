"""E2: bộ phân loại ảnh dùng Transformer encoder nông.

    ảnh -> tokenizer -> [CLS] + token -> + positional embedding
        -> L x EncoderBlock (pre-norm: x + MSA(LN(x)); x + FFN(LN(x)))
        -> LayerNorm -> lấy token CLS -> Linear -> 10 logits

Khối encoder tự lắp từ LayerNorm, MSA (scratch hoặc torch) và FFN, không dùng
nn.TransformerEncoderLayer, để hai bản chỉ khác nhau đúng ở phần MSA.
"""
import torch
import torch.nn as nn

from .msa import build_msa
from .tokenizers import build_tokenizer


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float,
                 attn_impl: str, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = build_msa(attn_impl, dim, num_heads, attn_dropout)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(self, tokenizer: str = "patch4", attn_impl: str = "scratch", embed_dim: int = 96,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1,
                 attn_dropout: float = 0.0, num_classes: int = 10):
        super().__init__()
        self.tokenizer = build_tokenizer(tokenizer, embed_dim)
        n = self.tokenizer.num_tokens
        self.num_tokens = n + 1  # + CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))  # học được
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_impl, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                                   # (B, 1, 28, 28)
        t = self.tokenizer(x)                               # (B, N, D)
        cls = self.cls_token.expand(t.size(0), -1, -1)      # (B, 1, D)
        t = torch.cat([cls, t], dim=1) + self.pos_embed     # (B, N+1, D)
        t = self.pos_drop(t)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return self.head(t[:, 0])                           # logits từ token CLS


@torch.no_grad()
def copy_weights_torch_to_scratch(scratch: ImageTransformer, ref: ImageTransformer) -> None:
    """Đưa model scratch về đúng trọng số của model torch (cùng kiến trúc).

    Các tham số ngoài MSA có cùng tên ở hai model nên chép thẳng; phần MSA chép bằng
    load_from_torch. Dùng để hai lần chạy (scratch / torch) xuất phát từ CÙNG trọng số,
    khi đó khác biệt kết quả chỉ còn do cách tính MSA.
    """
    ref_sd = ref.state_dict()
    for name, t in scratch.state_dict().items():
        if name in ref_sd and ref_sd[name].shape == t.shape:
            t.copy_(ref_sd[name])
    for s_blk, r_blk in zip(scratch.blocks, ref.blocks, strict=True):
        s_blk.attn.load_from_torch(r_blk.attn.mha)
