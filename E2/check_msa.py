"""Kiểm chứng MSA tự viết cho ra cùng kết quả với nn.MultiheadAttention.

Cách làm: tạo nn.MultiheadAttention ngẫu nhiên, chép trọng số sang bản scratch,
đưa cùng một đầu vào qua cả hai rồi so sánh
    (1) đầu ra, (2) ma trận attention từng head, (3) gradient theo đầu vào.
Chạy với nhiều cấu hình (D, h, N) và cả float32 lẫn float64.

    python -m E2.check_msa
Ghi kết quả vào results/E2/msa_check.json.
"""
import argparse

import torch
import torch.nn as nn

from common.utils import get_device, save_json, set_seed

from .msa import MultiHeadSelfAttentionScratch

CONFIGS = [  # (embed_dim, num_heads, num_tokens)
    (32, 4, 10),
    (96, 4, 50),
    (96, 8, 29),
    (64, 1, 17),
]


def compare(D, h, N, dtype, device, batch=8):
    ref = nn.MultiheadAttention(D, h, batch_first=True).to(device=device, dtype=dtype).eval()
    mine = MultiHeadSelfAttentionScratch(D, h).to(device=device, dtype=dtype).eval()
    mine.load_from_torch(ref)

    x1 = torch.randn(batch, N, D, device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    out_ref, attn_ref = ref(x1, x1, x1, need_weights=True, average_attn_weights=False)
    out_mine, attn_mine = mine(x2, return_attn=True)

    g = torch.randn_like(out_ref)  # cùng một gradient ở đầu ra cho cả hai
    out_ref.backward(g)
    out_mine.backward(g)

    return {
        "embed_dim": D, "num_heads": h, "num_tokens": N, "dtype": str(dtype).replace("torch.", ""),
        "max_abs_diff_output": (out_ref - out_mine).abs().max().item(),
        "max_abs_diff_attention": (attn_ref - attn_mine).abs().max().item(),
        "max_abs_diff_input_grad": (x1.grad - x2.grad).abs().max().item(),
        "attention_rows_sum_to_one": torch.allclose(attn_mine.sum(-1), torch.ones_like(attn_mine.sum(-1))),
        "params_scratch": sum(p.numel() for p in mine.parameters()),
        "params_torch": sum(p.numel() for p in ref.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E2/msa_check.json")
    args = ap.parse_args()
    set_seed(0)
    device = get_device()
    rows = []
    for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 1e-4)]:
        for D, h, N in CONFIGS:
            r = compare(D, h, N, dtype, device)
            worst = max(r["max_abs_diff_output"], r["max_abs_diff_attention"], r["max_abs_diff_input_grad"])
            r["tolerance"] = tol
            r["passed"] = bool(worst < tol and r["attention_rows_sum_to_one"]
                               and r["params_scratch"] == r["params_torch"])
            rows.append(r)
            print(f"D={D:3d} h={h} N={N:3d} {r['dtype']:8s} | out {r['max_abs_diff_output']:.2e} "
                  f"attn {r['max_abs_diff_attention']:.2e} grad {r['max_abs_diff_input_grad']:.2e} "
                  f"| params {r['params_scratch']} vs {r['params_torch']} | {'PASS' if r['passed'] else 'FAIL'}")
    save_json({"device": str(device), "results": rows}, args.out)
    if not all(r["passed"] for r in rows):
        raise SystemExit("Có cấu hình không khớp, xem bảng ở trên.")
    print(f"Tất cả khớp. Đã ghi {args.out}")


if __name__ == "__main__":
    main()
