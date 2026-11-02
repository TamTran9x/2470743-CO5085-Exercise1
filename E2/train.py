"""Huấn luyện E2.

Ví dụ:
    python -m E2.train --runs patch4:scratch patch4:torch row:scratch row:torch cnn:scratch patch7:scratch
    python -m E2.train --runs patch4:scratch --epochs 5
Kết quả ghi vào results/E2/<tokenizer>_<impl>/.

Với cùng tokenizer, bản scratch được khởi tạo bằng ĐÚNG trọng số của bản torch
(dựng model torch với cùng seed rồi chép sang), và hai bản dùng cùng thứ tự batch.
Vì vậy khác biệt giữa hai đường cong chỉ đến từ cách tính MSA.
"""
import argparse
import os

from common.data import get_data
from common.engine import TrainConfig, fit
from common.utils import get_device, set_seed

from .model import ImageTransformer, copy_weights_torch_to_scratch

DEFAULT_RUNS = ["patch4:scratch", "patch4:torch", "row:scratch", "row:torch", "cnn:scratch", "patch7:scratch"]


def build(tokenizer: str, impl: str, args, seed: int) -> ImageTransformer:
    kw = dict(tokenizer=tokenizer, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads,
              mlp_ratio=args.mlp_ratio, dropout=args.dropout, attn_dropout=0.0)
    set_seed(seed)
    ref = ImageTransformer(attn_impl="torch", **kw)
    if impl == "torch":
        return ref
    model = ImageTransformer(attn_impl="scratch", **kw)
    copy_weights_torch_to_scratch(model, ref)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="danh sách tokenizer:impl, impl thuộc {scratch, torch}")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/E2")
    args = ap.parse_args()

    device = get_device()
    for run in args.runs:
        tokenizer, impl = run.split(":")
        model = build(tokenizer, impl, args, args.seed)
        set_seed(args.seed)
        data = get_data(args.data_root, batch_size=args.batch_size, loader_seed=args.seed)
        cfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, seed=args.seed,
            extra={"exercise": "E2", "tokenizer": tokenizer, "attn_impl": impl,
                   "num_tokens": model.num_tokens, "embed_dim": args.embed_dim, "depth": args.depth,
                   "heads": args.heads, "mlp_ratio": args.mlp_ratio, "dropout": args.dropout,
                   "batch_size": args.batch_size},
        )
        fit(model, data, cfg, device, os.path.join(args.out, f"{tokenizer}_{impl}"),
            name=f"E2-{tokenizer}-{impl}")


if __name__ == "__main__":
    main()
