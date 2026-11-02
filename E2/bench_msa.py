"""Đo thời gian forward+backward của một lớp MSA (scratch vs torch) theo số token N.

Cho thấy chi phí attention tăng theo N^2: patch 7x7 (16 token), patch 4x4 (49),
patch chồng nhau (169), patch 2x2 (196), mỗi pixel một token (784).

    python -m E2.bench_msa
Ghi kết quả vào results/E2/msa_bench.json.
"""
import argparse

import torch
import torch.nn as nn

from common.utils import Timer, get_device, save_json, set_seed

from .msa import MultiHeadSelfAttentionScratch


def bench(module, x, iters, device):
    for _ in range(3):  # làm nóng
        module(x).sum().backward()
    with Timer(device) as t:
        for _ in range(iters):
            module(x).sum().backward()
    return t.elapsed / iters * 1000.0  # ms / lượt


def peak_mem_mb(module, x, device):
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    module(x).sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[16, 49, 169, 196, 784])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="results/E2/msa_bench.json")
    args = ap.parse_args()

    set_seed(0)
    device = get_device()
    iters = args.iters if device.type == "cuda" else max(2, args.iters // 10)
    ref = nn.MultiheadAttention(args.embed_dim, args.heads, batch_first=True).to(device)
    mine = MultiHeadSelfAttentionScratch(args.embed_dim, args.heads).to(device)
    mine.load_from_torch(ref)

    def torch_forward(x):
        return ref(x, x, x, need_weights=False)[0]

    class Wrap(nn.Module):  # để bench() gọi thống nhất module(x)
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    rows = []
    for n in args.tokens:
        x = torch.randn(args.batch, n, args.embed_dim, device=device, requires_grad=True)
        row = {"num_tokens": n, "batch": args.batch}
        for impl, mod in [("scratch", mine), ("torch", Wrap(torch_forward))]:
            try:
                row[f"{impl}_ms"] = bench(mod, x, iters, device)
                row[f"{impl}_peak_mem_mb"] = peak_mem_mb(mod, x, device)
            except torch.cuda.OutOfMemoryError:
                row[f"{impl}_ms"] = None
                row[f"{impl}_peak_mem_mb"] = None
                torch.cuda.empty_cache()
        # Bộ nhớ riêng cho ma trận attention của bản scratch: B * h * N * N * 4 byte
        row["attn_matrix_mb"] = args.batch * args.heads * n * n * 4 / 2 ** 20
        rows.append(row)
        print(row)
    save_json({"device": str(device), "embed_dim": args.embed_dim, "heads": args.heads,
               "iters": iters, "results": rows}, args.out)


if __name__ == "__main__":
    main()
