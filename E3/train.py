"""Huấn luyện E3.

Ví dụ:
    python -m E3.train --runs lstm:rows gru:rows lstm:patch7 gru:patch7
Kết quả ghi vào results/E3/<cell>_<seq>/.
Baseline MLP/CNN lấy từ E1 (cùng split, cùng seed): results/E1/mlp, results/E1/cnn.
"""
import argparse
import os

from common.data import get_data
from common.engine import TrainConfig, fit
from common.utils import get_device, set_seed

from .models import RNNClassifier

DEFAULT_RUNS = ["lstm:rows", "gru:rows", "lstm:patch7", "gru:patch7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="cell:seq_mode")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/E3")
    args = ap.parse_args()

    device = get_device()
    for run in args.runs:
        cell, seq_mode = run.split(":")
        set_seed(args.seed)
        data = get_data(args.data_root, batch_size=args.batch_size, loader_seed=args.seed)
        model = RNNClassifier(cell, seq_mode, args.hidden, args.layers, args.dropout)
        cfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, seed=args.seed,
            extra={"exercise": "E3", "cell": cell, "seq_mode": seq_mode, "seq_len": model.seq_len,
                   "input_size": model.input_size, "hidden": args.hidden, "layers": args.layers,
                   "dropout": args.dropout, "batch_size": args.batch_size},
        )
        fit(model, data, cfg, device, os.path.join(args.out, f"{cell}_{seq_mode}"),
            name=f"E3-{cell}-{seq_mode}")


if __name__ == "__main__":
    main()
