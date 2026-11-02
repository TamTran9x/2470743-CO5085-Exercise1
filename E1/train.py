"""Huấn luyện E1.

Ví dụ:
    python -m E1.train --model all
    python -m E1.train --model cnn --epochs 15
Kết quả ghi vào results/E1/<model>/.
"""
import argparse
import os

from common.data import get_data
from common.engine import TrainConfig, fit
from common.utils import get_device, set_seed

from .models import MODELS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/E1")
    args = ap.parse_args()

    device = get_device()
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        # Cùng seed, cùng split, cùng thứ tự batch, cùng optimizer cho cả ba mô hình -> so sánh công bằng.
        set_seed(args.seed)
        data = get_data(args.data_root, batch_size=args.batch_size, loader_seed=args.seed)
        model = build_model(name)
        cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                          seed=args.seed, extra={"exercise": "E1", "model": name,
                                                 "batch_size": args.batch_size})
        fit(model, data, cfg, device, os.path.join(args.out, name), name=f"E1-{name}")


if __name__ == "__main__":
    main()
