"""E3: mô hình dự đoán thế nào khi mới thấy một phần đầu của ảnh?

- LSTM/GRU: lấy logits ở từng bước t (RNN một chiều nên bước t chỉ phụ thuộc t phần
  đầu), không cần huấn luyện lại.
- MLP/CNN của E1: giữ k hàng đầu, các hàng còn lại thay bằng pixel đen, rồi dự đoán.

Không mô hình nào được huấn luyện với ảnh bị che, nên đây là phép thử "ngoài phân phối"
như nhau cho cả hai nhóm. Trục x là tỉ lệ số pixel đã thấy.

    python -m E3.temporal
Ghi kết quả vào results/E3/prefix_accuracy.json.
"""
import argparse
import os

import torch

from common.data import black_value, get_data
from common.utils import get_device, load_json, save_json
from E1.models import build_model

from .models import RNNClassifier


@torch.no_grad()
def rnn_prefix_accuracy(model, loader, device):
    model.eval()
    correct, n = None, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.logits_every_step(x)                      # (B, T, 10)
        hit = (logits.argmax(-1) == y[:, None]).sum(0).float()   # (T,)
        correct = hit if correct is None else correct + hit
        n += y.size(0)
    return (correct / n).cpu().tolist()


@torch.no_grad()
def masked_rows_accuracy(model, loader, device, fill: float, H: int = 28):
    model.eval()
    correct = torch.zeros(H, device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for k in range(1, H + 1):
            xm = x.clone()
            xm[:, :, k:, :] = fill                                # chỉ giữ k hàng đầu
            correct[k - 1] += (model(xm).argmax(1) == y).sum()
        n += y.size(0)
    return (correct / n).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/E1")
    ap.add_argument("--e3", default="results/E3")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    device = get_device()
    data = get_data(args.data_root)
    fill = black_value(data.mean, data.std)
    out = {}

    for run in sorted(os.listdir(args.e3)):
        run_dir = os.path.join(args.e3, run)
        ckpt = os.path.join(run_dir, "best.pt")
        if not os.path.isfile(ckpt):
            continue
        extra = load_json(os.path.join(run_dir, "summary.json"))["config"]["extra"]
        model = RNNClassifier(extra["cell"], extra["seq_mode"], extra["hidden"], extra["layers"],
                              extra["dropout"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        acc = rnn_prefix_accuracy(model, data.test, device)
        T = len(acc)
        out[run] = {"kind": "rnn", "fraction_seen": [(t + 1) / T for t in range(T)], "acc": acc}
        print(f"{run}: acc khi thấy 25/50/75/100% = "
              f"{acc[T // 4 - 1]:.3f} / {acc[T // 2 - 1]:.3f} / {acc[3 * T // 4 - 1]:.3f} / {acc[-1]:.3f}")

    for name in ["mlp", "cnn"]:
        ckpt = os.path.join(args.e1, name, "best.pt")
        if not os.path.isfile(ckpt):
            print(f"Bỏ qua {name}: chưa có {ckpt} (cần chạy E1 trước)")
            continue
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        acc = masked_rows_accuracy(model, data.test, device, fill)
        out[f"{name}_masked_rows"] = {"kind": "baseline", "fraction_seen": [(k + 1) / 28 for k in range(28)],
                                      "acc": acc}
        print(f"{name} (che hàng): acc khi thấy 7/14/21/28 hàng = "
              f"{acc[6]:.3f} / {acc[13]:.3f} / {acc[20]:.3f} / {acc[27]:.3f}")

    save_json(out, os.path.join(args.e3, "prefix_accuracy.json"))


if __name__ == "__main__":
    main()
