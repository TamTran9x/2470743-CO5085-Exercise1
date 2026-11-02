#!/usr/bin/env python3
"""Sinh 3 notebook độc lập E1/E1.ipynb, E2/E2.ipynb, E3/E3.ipynb từ các module trong repo.

Mỗi notebook tự chứa toàn bộ code (không import từ common/, E1/, ...), nên có thể tải riêng lên
Colab và chạy từ đầu đến cuối. Code của các phần dùng chung (dữ liệu, vòng lặp huấn luyện) và mô hình
được chép nguyên văn từ module tương ứng để notebook và script luôn giống nhau.

    python scripts/build_notebooks.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ tiện ích dựng notebook
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text, analysis=False):
    meta = {"tags": ["analysis"]} if analysis else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": _lines(text)}


def _lines(text):
    text = text.strip("\n")
    return text.splitlines(True)


def module_body(rel_path):
    """Nội dung module, bỏ docstring đầu file và các dòng import ở mức ngoài cùng."""
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r'^\s*""".*?"""\s*\n', "", src, count=1, flags=re.S)
    out = [ln for ln in src.splitlines() if not re.match(r"^(import |from )", ln)]
    return "\n".join(out).strip("\n")


def write_nb(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path, len(cells), "cells")


# ------------------------------------------------------------------ các ô dùng chung
SETUP = '''
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| thiết bị:",
      torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (sẽ chậm, nên bật GPU)")
'''

HELPERS = '''
# ---- hằng số và hàm vẽ/đọc kết quả (không cần PyTorch)
from matplotlib.ticker import MaxNLocator

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK2, MUTED, GRID = "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.8, "savefig.bbox": "tight",
})


def load_run(run_dir):
    """Đọc lại một lần chạy đã lưu: summary.json, history.csv, logits trên tập test."""
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    run = {"summary": summary, "history": pd.read_csv(os.path.join(run_dir, "history.csv"))}
    npz = os.path.join(run_dir, "test_outputs.npz")
    if os.path.isfile(npz):
        d = np.load(npz)
        run["logits"], run["targets"] = d["logits"], d["targets"]
        run["preds"] = run["logits"].argmax(1)
    return run


def summary_row(label, run):
    s = run["summary"]
    return {"Mô hình": label, "Số tham số": s["params"],
            "Train acc (%)": round(100 * s["final_train_acc"], 2),
            "Val acc tốt nhất (%)": round(100 * s["best_val_acc"], 2), "Epoch tốt nhất": s["best_epoch"],
            "Test acc (%)": round(100 * s["test_acc"], 2), "Test loss": round(s["test_loss"], 4),
            "Giây/epoch": round(s["mean_epoch_time"], 2), "Suy luận test (s)": round(s["test_inference_time"], 3)}


def plot_curves(runs, labels, colors, styles=None, save_as=None):
    """Loss và accuracy theo epoch. Nét chấm: train; nét liền/gạch: validation."""
    styles = styles or {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, run in runs.items():
        h, c, ls = run["history"], colors[key], styles.get(key, "-")
        ax1.plot(h["epoch"], h["train_loss"], color=c, ls=":", lw=1.3)
        ax1.plot(h["epoch"], h["val_loss"], color=c, ls=ls)
        ax2.plot(h["epoch"], 100 * h["train_acc"], color=c, ls=":", lw=1.3)
        ax2.plot(h["epoch"], 100 * h["val_acc"], color=c, ls=ls, label=labels[key])
    ax1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax2.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.5, -0.02, "nét chấm: train · nét liền/gạch: validation", ha="center", color=INK2)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    plt.show()


def confusion(targets, preds, k=10):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    return cm


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
'''

DATA_MD = '''
## Dữ liệu: Fashion-MNIST

- 60 000 ảnh train gốc được hoán vị bằng seed cố định: **54 000 train**, **6 000 validation**.
  10 000 ảnh test gốc chỉ dùng cho kết quả cuối.
- Ảnh đưa về $[0,1]$ rồi chuẩn hóa bằng mean/std tính **chỉ trên phần train**. Không tăng cường dữ liệu.
- Toàn bộ ảnh được giữ sẵn dưới dạng tensor trong bộ nhớ (28×28 nên rất nhẹ), mỗi epoch không phải đọc lại từ đĩa.
- Cả ba bài dùng đúng cùng split này (cùng seed), nên kết quả so sánh được với nhau.
'''

DATA_LOAD = '''
data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
print("Kích thước:", data.sizes, "| mean = %.4f, std = %.4f" % (data.mean, data.std))

# Ảnh test gốc (uint8, chưa chuẩn hóa) để hiển thị
TEST_IMAGES = datasets.FashionMNIST(DATA_ROOT, train=False, download=True).data.numpy()
TEST_LABELS = datasets.FashionMNIST(DATA_ROOT, train=False, download=True).targets.numpy()

fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for c in range(10):
    idx = np.where(TEST_LABELS == c)[0][:2]
    for r in range(2):
        axes[r, c].imshow(TEST_IMAGES[idx[r]], cmap="gray_r")
        axes[r, c].axis("off")
    axes[0, c].set_title(CLASSES[c], fontsize=8)
plt.suptitle("Hai ảnh mẫu của mỗi lớp")
plt.show()
'''

ENGINE_MD = '''
## Vòng lặp huấn luyện tự viết

Mỗi bước đi đúng trình tự **forward → loss → `backward()` → `optimizer.step()`**, không dùng `trainer.fit()`.

- **Loss:** `nn.CrossEntropyLoss` nhận logits (đã gồm log-softmax), nên mô hình không đặt softmax ở cuối `forward`.
- **Optimizer:** AdamW, không áp weight decay cho bias, tham số chuẩn hóa, token CLS và positional embedding.
- **Learning rate:** đổi theo từng bước, warmup tuyến tính (nếu có) rồi giảm theo cosine về 0.
- **Chọn mô hình:** sau mỗi epoch đánh giá trên validation, giữ checkpoint có val accuracy cao nhất;
  tập test chỉ được đánh giá một lần bằng checkpoint đó.
- **Lưu kết quả:** mỗi lần chạy ghi `history.csv`, `summary.json`, `best.pt`, `test_outputs.npz`
  (cùng định dạng với script trong repo, nên `scripts/make_report_assets.py` dùng lại được).
'''

ZIP_CELL = '''
# Nén kết quả để tải về (Colab). Chạy trên máy cá nhân thì kết quả đã nằm sẵn trong thư mục results/ và figures/.
import shutil

archive = shutil.make_archive(f"{EXERCISE}_results", "zip", ".", base_dir="results")
shutil.make_archive(f"{EXERCISE}_figures", "zip", ".", base_dir="figures")
print("Đã tạo", archive, "và", f"{EXERCISE}_figures.zip")
try:
    from google.colab import files
    files.download(archive)
    files.download(f"{EXERCISE}_figures.zip")
except ImportError:
    pass
'''


def config_cell(exercise, extra):
    return f'''
EXERCISE = "{exercise}"
SEED = 42
DATA_ROOT = "data"
RESULTS = "results/{exercise}"
FIGURES = "figures/{exercise}"
BATCH_SIZE = 128
QUICK = False          # True: mỗi mô hình chỉ 1 epoch, để chạy thử toàn bộ notebook thật nhanh
                       # (chạy thử xong nhớ xóa thư mục results/ trước khi chạy thật)
{extra.strip()}
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
'''


def common_cells(exercise, extra_config):
    return [
        md("## Thiết lập\nChạy lần lượt từ trên xuống. Trên Colab: *Runtime → Change runtime type → T4 GPU*."),
        code(SETUP),
        code(HELPERS, analysis=True),
        code(config_cell(exercise, extra_config), analysis=True),
        md("### Seed, đo thời gian, lưu file"),
        code(module_body("common/utils.py")),
        md(DATA_MD),
        code(module_body("common/data.py")),
        code(DATA_LOAD),
        md(ENGINE_MD),
        code(module_body("common/engine.py")),
    ]


# ================================================================== E1
def build_e1():
    cells = [md('''
# E1 — Phân loại ảnh: Softmax, MLP, CNN

**CO5085 · Học sâu và ứng dụng trong thị giác máy tính · HK261** — tập dữ liệu: Fashion-MNIST.

Notebook này tự chứa toàn bộ code của E1 và chạy được từ đầu đến cuối (khoảng 3 phút trên GPU T4).

| Yêu cầu của đề | Phần trong notebook |
|---|---|
| 1. Softmax classifier trên vector flatten | Mục *Mô hình* — `SoftmaxClassifier` |
| 2. MLP nhiều tầng trên vector flatten | `MLPClassifier` |
| 3. CNN (conv/pool/BN + head) | `CNNClassifier` |
| 4. Tự viết training loop, đánh giá val/test | Mục *Vòng lặp huấn luyện* |
| 5. So sánh accuracy, số tham số, đường cong, lỗi | Mục *Kết quả* và *Phân tích lỗi* |
''')]
    cells += common_cells("E1", '''
EPOCHS = 1 if QUICK else 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
E1_NAMES = ["softmax", "mlp", "cnn"]
LABEL = {"softmax": "Softmax", "mlp": "MLP", "cnn": "CNN"}
COLOR = {"softmax": SERIES[0], "mlp": SERIES[1], "cnn": SERIES[2]}
''')
    cells += [
        md('''
## Mô hình

- **Softmax:** flatten → `Linear(784, 10)`. Hồi quy logistic đa lớp; ranh giới giữa hai lớp là một siêu phẳng.
- **MLP:** flatten → 784 → 512 → 256 → 10, ReLU + Dropout. Phi tuyến, nhưng vẫn không biết hai pixel nào nằm cạnh nhau.
- **CNN:** hai khối [conv3×3–BN–ReLU]×2 + maxpool (28→14→7), rồi FC 3136 → 128 → 10.
  Conv để `bias=False` vì BatchNorm ngay sau đã có tham số dịch.
  Kích thước được chọn để CNN và MLP có số tham số cùng cỡ.

Cả ba trả về **logits**; softmax nằm trong `CrossEntropyLoss` khi huấn luyện.
'''),
        code(module_body("E1/models.py")),
        md("### Số tham số theo từng tầng"),
        code('''
def param_table(model):
    rows = [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()]
    return pd.DataFrame(rows, columns=["tham số", "shape", "số phần tử"])


x_demo = torch.randn(2, 1, 28, 28)
for name in E1_NAMES:
    m = build_model(name)
    print(f"{LABEL[name]}: {count_params(m):,} tham số | output {tuple(m(x_demo).shape)}")
    display(param_table(m))
'''),
        md('''
## Huấn luyện

Cả ba mô hình dùng **cùng** seed, split, thứ tự batch, optimizer (AdamW, LR 1e-3, weight decay 1e-4),
cosine schedule và số epoch, để khác biệt kết quả chỉ đến từ kiến trúc.
'''),
        code('''
for name in E1_NAMES:
    set_seed(SEED)
    data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
    model = build_model(name)
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SEED,
                      extra={"exercise": "E1", "model": name, "batch_size": BATCH_SIZE})
    fit(model, data, cfg, device, os.path.join(RESULTS, name), name=f"E1-{name}")
'''),
        md("## Kết quả\nĐọc lại các lần chạy đã lưu (chạy được độc lập với ô huấn luyện nếu đã có `results/E1/`)."),
        code('''
runs = {n: load_run(os.path.join(RESULTS, n)) for n in E1_NAMES}
table = pd.DataFrame([summary_row(LABEL[n], runs[n]) for n in E1_NAMES]).set_index("Mô hình")
display(table)
table.to_csv(os.path.join(FIGURES, "e1_table.csv"))
''', analysis=True),
        md('''
### Đường cong học
Khoảng cách giữa nét chấm (train) và nét liền (validation) cho biết mức overfit.
Để ý val loss của CNN: nó có thể tăng lại trong khi val accuracy vẫn tăng.
'''),
        code('''
plot_curves(runs, LABEL, COLOR, save_as=os.path.join(FIGURES, "e1_curves.png"))
for n in E1_NAMES:
    h = runs[n]["history"]
    i = int(h["val_loss"].idxmin())
    print(f"{LABEL[n]:8s} val loss thấp nhất {h['val_loss'][i]:.4f} ở epoch {int(h['epoch'][i])}, "
          f"cuối cùng {h['val_loss'].iloc[-1]:.4f} | train loss cuối {h['train_loss'].iloc[-1]:.4f}")
''', analysis=True),
        md("## Phân tích lỗi\n### Accuracy theo lớp và confusion matrix"),
        code('''
cms = {n: confusion(runs[n]["targets"], runs[n]["preds"]) for n in E1_NAMES}
per_class = pd.DataFrame({LABEL[n]: (100 * np.diag(cms[n]) / cms[n].sum(1)).round(2) for n in E1_NAMES},
                         index=CLASSES)
display(per_class)

# Confusion matrix của CNN: số trong ô là số ảnh, màu là tỉ lệ theo hàng
cm = cms["cnn"]
norm = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "black")
ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
ax.set_yticks(range(10), CLASSES)
ax.set(xlabel="lớp dự đoán", ylabel="lớp thật", title="Confusion matrix — CNN (tập test)")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(FIGURES, "e1_confusion_cnn.png"))
plt.show()

# Ba mô hình, chỉ phần lỗi (bỏ đường chéo), cùng thang màu
errs = {n: cms[n] / cms[n].sum(1, keepdims=True) * (1 - np.eye(10)) for n in E1_NAMES}
vmax = max(e.max() for e in errs.values())
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, n in zip(axes, E1_NAMES, strict=True):
    im = ax.imshow(errs[n], cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(10), SHORT, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10), SHORT, fontsize=7)
    ax.set_title(f"{LABEL[n]} — test acc {100 * runs[n]['summary']['test_acc']:.2f}%")
    ax.grid(False)
fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02, label="tỉ lệ lỗi theo hàng")
fig.savefig(os.path.join(FIGURES, "e1_confusion_all.png"))
plt.show()
''', analysis=True),
        md("### Các cặp lớp hay nhầm và mức chồng lấp lỗi giữa ba mô hình"),
        code('''
off = cms["cnn"].copy()
np.fill_diagonal(off, 0)
top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:6], off.shape))[0]
pairs = pd.DataFrame([{"Nhãn thật": CLASSES[t], "Bị đoán là": CLASSES[p],
                       **{LABEL[n]: int(cms[n][t, p]) for n in E1_NAMES}} for t, p in top])
display(pairs)

UPPER = [0, 2, 4, 6]  # T-shirt/top, Pullover, Coat, Shirt
for n in E1_NAMES:
    cm_n = cms[n]
    total = cm_n.sum() - np.trace(cm_n)
    inside = sum(cm_n[a, b] for a in UPPER for b in UPPER if a != b)
    print(f"{LABEL[n]:8s} {total:5d} ảnh sai, {100 * inside / total:.1f}% nằm trong nhóm áo thân trên")

wrong = {n: runs[n]["preds"] != runs[n]["targets"] for n in E1_NAMES}
print(f"{100 * (wrong['cnn'] & wrong['mlp']).sum() / wrong['cnn'].sum():.1f}% ảnh CNN sai cũng bị MLP sai")
print(f"{(wrong['cnn'] & wrong['mlp'] & wrong['softmax']).sum()} ảnh cả ba mô hình đều sai")
''', analysis=True),
        md("### Những ảnh CNN đoán sai với độ tự tin cao nhất\nNhiều ảnh trong số này mơ hồ ngay cả với người, có ảnh có vẻ bị gán nhãn sai."),
        code('''
prob = softmax_np(runs["cnn"]["logits"])
tg, pr = runs["cnn"]["targets"], runs["cnn"]["preds"]
wrong_idx = np.where(pr != tg)[0]
order = wrong_idx[np.argsort(-prob[wrong_idx, pr[wrong_idx]])][:16]
fig, axes = plt.subplots(2, 8, figsize=(13, 4.6))
for ax, i in zip(axes.flat, order, strict=False):
    ax.imshow(TEST_IMAGES[i], cmap="gray_r")
    ax.set_title(f"thật: {CLASSES[tg[i]]}\\nđoán: {CLASSES[pr[i]]}\\np={prob[i, pr[i]]:.2f}", fontsize=7)
    ax.axis("off")
fig.tight_layout(h_pad=1.5)
fig.savefig(os.path.join(FIGURES, "e1_errors.png"))
plt.show()
''', analysis=True),
        md('''
## Ghi chú khi viết nhận xét

- So sánh accuracy **và** số tham số: CNN ít tham số hơn MLP nhưng chính xác hơn — vì sao (tính cục bộ, chia sẻ trọng số)?
- Đường cong: mô hình nào underfit, mô hình nào overfit; val loss tăng trong khi val accuracy vẫn tăng nghĩa là gì?
- Lỗi tập trung ở lớp nào, cặp lớp nào; các ảnh sai tự tin cho biết gì về dữ liệu.

## Lưu kết quả
'''),
        code(ZIP_CELL),
    ]
    write_nb("E1/E1.ipynb", cells)


# ================================================================== E2
def build_e2():
    cells = [md('''
# E2 — Transformer: Multi-Head Self-Attention và tokenization

**CO5085 · Học sâu và ứng dụng trong thị giác máy tính · HK261** — tập dữ liệu: Fashion-MNIST (cùng split với E1).

Notebook này tự chứa toàn bộ code của E2 và chạy được từ đầu đến cuối (khoảng 15–20 phút trên GPU T4).

| Yêu cầu của đề | Phần trong notebook |
|---|---|
| 1. MSA tự viết (Linear, matmul, softmax, ghép head) | `MultiHeadSelfAttentionScratch` |
| 2. MSA bằng PyTorch để đối chiếu | `MultiHeadSelfAttentionTorch` (`nn.MultiheadAttention`) |
| 3. Đưa MSA vào bộ phân loại ảnh (encoder nông + head) | `ImageTransformer` |
| 4. Ít nhất hai cách lấy token | patch 4×4, patch 7×7, theo hàng, CNN-stem |
| 5. So sánh tự viết vs PyTorch; so sánh các tokenizer | Mục *Kết quả* |
''')]
    cells += common_cells("E2", '''
EPOCHS = 1 if QUICK else 20
LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1.0
EMBED_DIM, DEPTH, HEADS, MLP_RATIO, DROPOUT = 96, 4, 4, 2.0, 0.1
# Mỗi lần chạy là "tokenizer:cài_đặt_MSA". Thêm "patch4s2:scratch" để thử patch chồng nhau (169 token, chậm hơn).
RUNS = ["patch4:scratch", "patch4:torch", "row:scratch", "row:torch", "cnn:scratch", "patch7:scratch"]
TOK_LABEL = {"patch4": "Patch 4×4", "patch7": "Patch 7×7", "patch4s2": "Patch 4×4 chồng",
             "row": "Theo hàng", "cnn": "CNN-stem"}
IMPL_LABEL = {"scratch": "tự viết", "torch": "PyTorch"}
''')
    cells += [
        md(r'''
## 1. Multi-Head Self-Attention tự viết

Với $X\in\mathbb{R}^{N\times D}$, $h$ head, $d_k = D/h$:

$$Q = XW_Q,\; K = XW_K,\; V = XW_V,\qquad
A_i = \mathrm{softmax}\!\Big(\frac{Q_iK_i^\top}{\sqrt{d_k}}\Big),\qquad
\mathrm{MSA}(X) = \mathrm{concat}(A_1V_1,\dots,A_hV_h)\,W_O$$

Theo dõi shape: `(B, N, D)` → chiếu Q, K, V → tách head `(B, h, N, d_k)` → điểm `(B, h, N, N)` → softmax theo
chiều cuối → nhân V → ghép head về `(B, N, D)` → `out_proj`.
Bản tự viết chỉ dùng `nn.Linear`, `torch.matmul`, `torch.softmax` và đổi shape; bản đối chiếu bọc `nn.MultiheadAttention`.
'''),
        code(module_body("E2/msa.py")),
        md('''
### Kiểm chứng: MSA tự viết == `nn.MultiheadAttention`

Tạo `nn.MultiheadAttention` ngẫu nhiên, chép trọng số sang bản tự viết (`in_proj_weight` là $[W_Q; W_K; W_V]$
xếp chồng), đưa cùng đầu vào qua cả hai rồi so **đầu ra**, **ma trận attention từng head** và **gradient theo đầu vào**,
với cả float64 và float32.
'''),
        code('''
def compare_msa(D, h, N, dtype, batch=8):
    ref = nn.MultiheadAttention(D, h, batch_first=True).to(device=device, dtype=dtype).eval()
    mine = MultiHeadSelfAttentionScratch(D, h).to(device=device, dtype=dtype).eval()
    mine.load_from_torch(ref)
    x1 = torch.randn(batch, N, D, device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out_ref, attn_ref = ref(x1, x1, x1, need_weights=True, average_attn_weights=False)
    out_mine, attn_mine = mine(x2, return_attn=True)
    g = torch.randn_like(out_ref)       # cùng một gradient ở đầu ra cho cả hai
    out_ref.backward(g)
    out_mine.backward(g)
    return {"embed_dim": D, "num_heads": h, "num_tokens": N, "dtype": str(dtype).replace("torch.", ""),
            "max_abs_diff_output": (out_ref - out_mine).abs().max().item(),
            "max_abs_diff_attention": (attn_ref - attn_mine).abs().max().item(),
            "max_abs_diff_input_grad": (x1.grad - x2.grad).abs().max().item(),
            "attention_rows_sum_to_one": bool(torch.allclose(attn_mine.sum(-1), torch.ones_like(attn_mine.sum(-1)))),
            "params_scratch": sum(p.numel() for p in mine.parameters()),
            "params_torch": sum(p.numel() for p in ref.parameters())}


set_seed(0)
check_rows = []
for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 1e-4)]:
    for D, h, N in [(32, 4, 10), (96, 4, 50), (96, 8, 29), (64, 1, 17)]:
        r = compare_msa(D, h, N, dtype)
        worst = max(r["max_abs_diff_output"], r["max_abs_diff_attention"], r["max_abs_diff_input_grad"])
        r["tolerance"] = tol
        r["passed"] = bool(worst < tol and r["attention_rows_sum_to_one"] and r["params_scratch"] == r["params_torch"])
        check_rows.append(r)
save_json({"device": str(device), "results": check_rows}, os.path.join(RESULTS, "msa_check.json"))
assert all(r["passed"] for r in check_rows), "MSA tự viết không khớp nn.MultiheadAttention"
print("MSA tự viết khớp nn.MultiheadAttention ở mọi cấu hình.")
'''),
        code('''
with open(os.path.join(RESULTS, "msa_check.json"), encoding="utf-8") as f:
    check = pd.DataFrame(json.load(f)["results"])
display(check[["embed_dim", "num_heads", "num_tokens", "dtype", "max_abs_diff_output",
               "max_abs_diff_attention", "max_abs_diff_input_grad", "params_scratch", "params_torch", "passed"]])
''', analysis=True),
        md('''
### Chi phí một lớp MSA theo số token $N$

Forward + backward một lớp MSA với batch 128, $D=96$, $h=4$. $N$ = 16 (patch 7×7), 49 (patch 4×4),
169 (patch chồng), 196 (patch 2×2), 784 (mỗi pixel một token). Ma trận attention có $B\\cdot h\\cdot N^2$ phần tử.
'''),
        code('''
def bench_ms(fn, x, iters):
    for _ in range(3):                       # làm nóng
        fn(x).sum().backward()
    with Timer(device) as t:
        for _ in range(iters):
            fn(x).sum().backward()
    return t.elapsed / iters * 1000.0


def peak_mem_mb(fn, x):
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    fn(x).sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


set_seed(0)
B_BENCH, ITERS = 128, (20 if device.type == "cuda" else 2)
ref_mha = nn.MultiheadAttention(EMBED_DIM, HEADS, batch_first=True).to(device)
mine_msa = MultiHeadSelfAttentionScratch(EMBED_DIM, HEADS).to(device)
mine_msa.load_from_torch(ref_mha)
impls = {"scratch": lambda x: mine_msa(x), "torch": lambda x: ref_mha(x, x, x, need_weights=False)[0]}

bench_rows = []
for n in [16, 49, 169, 196, 784]:
    x = torch.randn(B_BENCH, n, EMBED_DIM, device=device, requires_grad=True)
    row = {"num_tokens": n, "batch": B_BENCH, "attn_matrix_mb": B_BENCH * HEADS * n * n * 4 / 2 ** 20}
    for impl, fn in impls.items():
        try:
            row[f"{impl}_ms"] = bench_ms(fn, x, ITERS)
            row[f"{impl}_peak_mem_mb"] = peak_mem_mb(fn, x)
        except torch.cuda.OutOfMemoryError:
            row[f"{impl}_ms"] = row[f"{impl}_peak_mem_mb"] = None
            torch.cuda.empty_cache()
    bench_rows.append(row)
    print(row)
save_json({"device": str(device), "embed_dim": EMBED_DIM, "heads": HEADS, "iters": ITERS, "results": bench_rows},
          os.path.join(RESULTS, "msa_bench.json"))
'''),
        code('''
with open(os.path.join(RESULTS, "msa_bench.json"), encoding="utf-8") as f:
    bench = pd.DataFrame(json.load(f)["results"])
display(bench.round(2))
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for impl, c in [("scratch", SERIES[0]), ("torch", SERIES[1])]:
    ok = bench[f"{impl}_ms"].notna()
    ax.plot(bench["num_tokens"][ok], bench[f"{impl}_ms"][ok], marker="o", ms=5, color=c, label=f"MSA {IMPL_LABEL[impl]}")
ax.set(xscale="log", yscale="log", xlabel="số token N", ylabel="ms / lượt (forward + backward)",
       title="Chi phí một lớp MSA theo số token")
ax.set_xticks([16, 49, 196, 784], ["16", "49", "196", "784"])
ax.minorticks_off()
ax.legend()
fig.savefig(os.path.join(FIGURES, "e2_bench.png"))
plt.show()
''', analysis=True),
        md('''
## 2. Các cách lấy token

| Tokenizer | Mỗi token là | Số token |
|---|---|---|
| `patch4` | ô 4×4 không chồng nhau, chiếu tuyến tính 16 → D | 49 |
| `patch7` | ô 7×7 không chồng nhau, chiếu tuyến tính 49 → D | 16 |
| `patch4s2` | ô 4×4 với stride 2 (chồng nhau) | 169 |
| `row` | một hàng 28 pixel, chiếu tuyến tính 28 → D | 28 |
| `cnn` | một vị trí trên feature map 7×7 của CNN 2 tầng + conv 1×1 về D kênh | 49 |

Patch embedding được viết bằng `nn.Conv2d(kernel_size = stride = p)`. Ô dưới chứng minh nó **tương đương**
cắt ảnh thành ô p×p, flatten rồi nhân cùng một ma trận chiếu.
'''),
        code(module_body("E2/tokenizers.py")),
        code('''
x_demo = torch.randn(2, 1, 28, 28)
for name in ["patch4", "patch7", "patch4s2", "row", "cnn"]:
    tok = build_tokenizer(name, EMBED_DIM)
    print(f"{name:9s} -> token {tuple(tok(x_demo).shape)}  (num_tokens = {tok.num_tokens})")

# Conv2d(k = s = p) == patchify + Linear
tok = PatchTokenizer(32, patch_size=4)
lin = nn.Linear(16, 32)
with torch.no_grad():
    lin.weight.copy_(tok.proj.weight.reshape(32, -1))
    lin.bias.copy_(tok.proj.bias)
print("Conv2d patch embedding == patchify + Linear:",
      torch.allclose(tok(x_demo), lin(patchify(x_demo, 4)), atol=1e-5))
'''),
        md(r'''
## 3. Bộ phân loại: Transformer encoder nông

ảnh → tokenizer → thêm token **CLS** → cộng **positional embedding** (học được) → 4 khối encoder → LayerNorm → Linear trên CLS.

Mỗi khối dạng *pre-norm*: $X \leftarrow X + \mathrm{MSA}(\mathrm{LN}(X))$, $X \leftarrow X + \mathrm{FFN}(\mathrm{LN}(X))$,
với FFN = Linear–GELU–Linear. Khối encoder được tự lắp (không dùng `nn.TransformerEncoderLayer`) để bản tự viết
và bản PyTorch **chỉ khác nhau đúng ở phần MSA**.
'''),
        code(module_body("E2/model.py")),
        code('''
# Sau khi chép trọng số, model tự viết và model PyTorch phải cho cùng output (float64)
x64 = torch.randn(3, 1, 28, 28, dtype=torch.float64)
for tok in ["patch4", "row", "cnn"]:
    kw = dict(tokenizer=tok, embed_dim=32, depth=2, num_heads=4)
    ref_model = ImageTransformer(attn_impl="torch", **kw).double().eval()
    my_model = ImageTransformer(attn_impl="scratch", **kw).double().eval()
    copy_weights_torch_to_scratch(my_model, ref_model)
    diff = (ref_model(x64) - my_model(x64)).abs().max().item()
    print(f"{tok:7s} lệch lớn nhất {diff:.1e} | tham số {count_params(my_model):,} vs {count_params(ref_model):,}")
for tok in ["patch4", "patch7", "row", "cnn"]:
    m = ImageTransformer(tok, "scratch", EMBED_DIM, DEPTH, HEADS, MLP_RATIO, DROPOUT)
    print(f"Cấu hình huấn luyện, tokenizer {tok:7s}: {m.num_tokens} token (gồm CLS), {count_params(m):,} tham số")
'''),
        md('''
## 4. Huấn luyện

- Cùng tokenizer: bản **tự viết** được khởi tạo bằng **đúng trọng số** của bản PyTorch (dựng model PyTorch với seed
  cố định rồi chép sang) và dùng cùng thứ tự batch, dropout attention = 0. Khác biệt kết quả chỉ còn đến từ cách tính MSA.
- Giữa các tokenizer: cùng encoder và siêu tham số — AdamW (LR 1e-3, weight decay 0.05), warmup 1 epoch rồi cosine.
'''),
        code('''
def build_transformer(tokenizer, impl, seed=SEED):
    kw = dict(tokenizer=tokenizer, embed_dim=EMBED_DIM, depth=DEPTH, num_heads=HEADS,
              mlp_ratio=MLP_RATIO, dropout=DROPOUT, attn_dropout=0.0)
    set_seed(seed)
    ref = ImageTransformer(attn_impl="torch", **kw)
    if impl == "torch":
        return ref
    model = ImageTransformer(attn_impl="scratch", **kw)
    copy_weights_torch_to_scratch(model, ref)
    return model


for run in RUNS:
    tokenizer, impl = run.split(":")
    model = build_transformer(tokenizer, impl)
    set_seed(SEED)
    data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS, seed=SEED,
                      extra={"exercise": "E2", "tokenizer": tokenizer, "attn_impl": impl,
                             "num_tokens": model.num_tokens, "embed_dim": EMBED_DIM, "depth": DEPTH,
                             "heads": HEADS, "mlp_ratio": MLP_RATIO, "dropout": DROPOUT, "batch_size": BATCH_SIZE})
    fit(model, data, cfg, device, os.path.join(RESULTS, f"{tokenizer}_{impl}"), name=f"E2-{tokenizer}-{impl}")
'''),
        md("## 5. Kết quả"),
        code('''
runs = {}
for run in RUNS:
    tok, impl = run.split(":")
    runs[(tok, impl)] = load_run(os.path.join(RESULTS, f"{tok}_{impl}"))
labels = {k: f"{TOK_LABEL[k[0]]} · {IMPL_LABEL[k[1]]}" for k in runs}
table = pd.DataFrame([{**summary_row(labels[k], r), "Số token": r["summary"]["config"]["extra"]["num_tokens"]}
                      for k, r in runs.items()]).set_index("Mô hình")
display(table)
table.to_csv(os.path.join(FIGURES, "e2_table.csv"))
''', analysis=True),
        md("### MSA tự viết vs PyTorch (cùng tokenizer, cùng khởi tạo)"),
        code('''
rows = []
for tok in TOK_LABEL:
    if (tok, "scratch") in runs and (tok, "torch") in runs:
        a, b = runs[(tok, "scratch")], runs[(tok, "torch")]
        sa, sb = a["summary"], b["summary"]
        rows.append({"Tokenizer": TOK_LABEL[tok],
                     "Test acc tự viết (%)": round(100 * sa["test_acc"], 2),
                     "Test acc PyTorch (%)": round(100 * sb["test_acc"], 2),
                     "Val acc lệch tối đa qua các epoch": round(100 * (a["history"]["val_acc"]
                                                                       - b["history"]["val_acc"]).abs().max(), 2),
                     "Giây/epoch tự viết": round(sa["mean_epoch_time"], 2),
                     "Giây/epoch PyTorch": round(sb["mean_epoch_time"], 2),
                     "Suy luận tự viết (s)": round(sa["test_inference_time"], 3),
                     "Suy luận PyTorch (s)": round(sb["test_inference_time"], 3)})
display(pd.DataFrame(rows).set_index("Tokenizer"))
''', analysis=True),
        md("### Đường cong học\nMàu: tokenizer; nét liền: MSA tự viết, nét gạch: PyTorch."),
        code('''
tok_order = [t for t in TOK_LABEL if any(k[0] == t for k in runs)]
colors = {k: SERIES[tok_order.index(k[0])] for k in runs}
styles = {k: "-" if k[1] == "scratch" else "--" for k in runs}
plot_curves(runs, labels, colors, styles, save_as=os.path.join(FIGURES, "e2_curves.png"))
''', analysis=True),
        md('''
### Attention của token CLS (khối encoder cuối, patch 4×4, MSA tự viết)

Trung bình theo head, mức chú ý từ token CLS tới 49 patch, phóng về 28×28 và phủ lên ảnh.
Vùng sáng là các patch mô hình dựa vào nhiều nhất để phân loại.
'''),
        code('''
@torch.no_grad()
def cls_attention_map(model, x):
    model.eval()
    t = model.tokenizer(x)
    cls = model.cls_token.expand(t.size(0), -1, -1)
    t = torch.cat([cls, t], dim=1) + model.pos_embed
    for blk in model.blocks[:-1]:
        t = blk(t)
    last = model.blocks[-1]
    _, attn = last.attn(last.norm1(t), return_attn=True)   # (B, h, N+1, N+1)
    a = attn[:, :, 0, 1:].mean(1)                          # CLS -> các patch, trung bình theo head
    side = model.tokenizer.grid[0]
    return a.reshape(-1, side, side).cpu().numpy()


ckpt = os.path.join(RESULTS, "patch4_scratch", "best.pt")
if os.path.isfile(ckpt):
    vit = build_transformer("patch4", "scratch").to(device)
    vit.load_state_dict(torch.load(ckpt, map_location=device))
    data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
    x_test, y_test = next(iter(data.test))
    pick = [int(np.where(y_test.numpy() == c)[0][0]) for c in range(10)]
    maps = cls_attention_map(vit, x_test[pick].to(device))
    with torch.no_grad():
        preds = vit(x_test[pick].to(device)).argmax(1).cpu().numpy()
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.3))
    for j, i in enumerate(pick):
        axes[0, j].imshow(TEST_IMAGES[i], cmap="gray_r")
        axes[0, j].set_title(f"{CLASSES[int(y_test[i])]}\\nđoán: {CLASSES[int(preds[j])]}", fontsize=7)
        up = np.kron(maps[j], np.ones((4, 4)))
        axes[1, j].imshow(TEST_IMAGES[i], cmap="gray_r")
        axes[1, j].imshow(up, cmap="inferno", alpha=0.6)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.savefig(os.path.join(FIGURES, "e2_cls_attention.png"))
    plt.show()
else:
    print("Chưa có checkpoint patch4_scratch — cần chạy ô huấn luyện với 'patch4:scratch' trong RUNS.")
'''),
        md('''
## Ghi chú khi viết nhận xét

- Tự viết vs PyTorch: accuracy và đường cong có trùng nhau không; khác biệt thời gian/bộ nhớ nằm ở đâu (xem benchmark theo N).
- Tokenizer: số token ảnh hưởng thế nào đến accuracy và thời gian; vì sao CNN-stem khác patch.
- So với CNN của E1: Transformer có vượt được không, vì sao.

## Lưu kết quả
'''),
        code(ZIP_CELL),
    ]
    write_nb("E2/E2.ipynb", cells)


# ================================================================== E3
def build_e3():
    cells = [md('''
# E3 — LSTM / GRU trên chuỗi ảnh

**CO5085 · Học sâu và ứng dụng trong thị giác máy tính · HK261** — tập dữ liệu: Fashion-MNIST (cùng split với E1, E2).

Notebook này tự chứa toàn bộ code của E3 và chạy được từ đầu đến cuối (khoảng 5 phút trên GPU T4).

| Yêu cầu của đề | Phần trong notebook |
|---|---|
| 1. LSTM và GRU (`nn.LSTM`, `nn.GRU`) cho phân loại | `RNNClassifier` |
| 2. Cách đưa ảnh thành chuỗi (≥ 2 cách) | theo hàng (28 bước) và theo patch 7×7 (16 bước) |
| 3. Baseline không hồi quy MLP/CNN | MLP, CNN của E1 (dùng lại checkpoint hoặc huấn luyện lại cùng split) |
| 4. So sánh accuracy, số tham số, khả năng bắt phụ thuộc thời gian | Mục *Kết quả* và *Nhìn một phần ảnh* |
''')]
    cells += common_cells("E3", '''
EPOCHS = 1 if QUICK else 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN, LAYERS, RNN_DROPOUT, GRAD_CLIP = 128, 2, 0.2, 1.0
RUNS = ["lstm:rows", "gru:rows", "lstm:patch7", "gru:patch7"]   # cell:cách biểu diễn
E1_RESULTS = "results/E1"     # baseline MLP/CNN: dùng lại nếu đã chạy E1, nếu chưa thì huấn luyện tại đây
CELL_LABEL = {"lstm": "LSTM", "gru": "GRU"}
SEQ_LABEL = {"rows": "theo hàng", "cols": "theo cột", "patch7": "patch 7×7", "patch4": "patch 4×4"}
''')
    cells += [
        md('''
## 1. Biến ảnh thành chuỗi

- **Theo hàng (`rows`):** bước thời gian $t$ là hàng thứ $t$, đọc từ trên xuống — $T=28$ bước, mỗi bước $F=28$ giá trị.
- **Theo patch (`patch7`):** lưới 4×4 patch 7×7, duyệt theo hàng (trái → phải, trên → dưới) — $T=16$, $F=49$.

Hai cách đánh đổi độ dài chuỗi với lượng thông tin mỗi bước.
'''),
        code(module_body("E3/models.py")),
        code('''
img = datasets.FashionMNIST(DATA_ROOT, train=False, download=True).data[:1].float().unsqueeze(1) / 255.0
seq_rows = to_sequence(img, "rows")[0].numpy()        # (28, 28)
seq_patch = to_sequence(img, "patch7")[0].numpy()     # (16, 49)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.9]})
axes[0].imshow(img[0, 0].numpy(), cmap="gray_r")
axes[0].set_title("ảnh gốc 28×28")
axes[1].imshow(seq_rows, cmap="gray_r", aspect="auto")
axes[1].set(title="theo hàng: T=28, F=28", xlabel="đặc trưng F", ylabel="bước t")
axes[2].imshow(seq_patch, cmap="gray_r", aspect="auto")
axes[2].set(title="theo patch 7×7: T=16, F=49", xlabel="đặc trưng F (7×7 pixel)", ylabel="bước t")
for ax in axes:
    ax.grid(False)
plt.tight_layout()
plt.show()
'''),
        md(r'''
## 2. Mô hình LSTM / GRU

chuỗi `(B, T, F)` → `nn.LSTM` hoặc `nn.GRU` (2 tầng, hidden 128, dropout 0.2 giữa các tầng, `batch_first=True`)
→ hidden của **bước cuối** ở tầng trên cùng → `Linear(128, 10)`.

Số tham số tầng đầu: $g\,(HF + H^2 + 2H)$ với $g=4$ cổng cho LSTM (input, forget, cell, output) và $g=3$ cho GRU
(reset, update, new) — cùng $H$ thì GRU có đúng 3/4 tham số hồi quy của LSTM.
'''),
        code('''
rows = []
for run in RUNS:
    cell, mode = run.split(":")
    m = RNNClassifier(cell, mode, HIDDEN, LAYERS, RNN_DROPOUT)
    rows.append({"Mô hình": f"{CELL_LABEL[cell]} {SEQ_LABEL[mode]}", "T": m.seq_len, "F": m.input_size,
                 "Tham số RNN (đếm)": count_params(m.rnn),
                 "Tham số RNN (công thức)": rnn_param_formula(cell, m.input_size, HIDDEN, LAYERS),
                 "Tổng tham số": count_params(m)})
display(pd.DataFrame(rows).set_index("Mô hình"))
'''),
        md('''
## 3. Baseline không hồi quy: MLP và CNN của E1

Nếu đã chạy notebook E1 (có `results/E1/mlp` và `results/E1/cnn`), dùng lại checkpoint. Nếu chưa, huấn luyện lại
đúng cấu hình E1 trên cùng split.
'''),
        code(module_body("E1/models.py")),
        code('''
for name in ["mlp", "cnn"]:
    out_dir = os.path.join(E1_RESULTS, name)
    if os.path.isfile(os.path.join(out_dir, "best.pt")) and os.path.isfile(os.path.join(out_dir, "summary.json")):
        print(f"Dùng lại {out_dir}")
        continue
    set_seed(SEED)
    data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
    cfg = TrainConfig(epochs=1 if QUICK else 15, lr=1e-3, weight_decay=1e-4, seed=SEED,
                      extra={"exercise": "E1", "model": name, "batch_size": BATCH_SIZE})
    fit(build_model(name), data, cfg, device, out_dir, name=f"E1-{name}")
'''),
        md("## 4. Huấn luyện LSTM / GRU\nAdamW (LR 1e-3, weight decay 1e-4), cosine, cắt gradient theo chuẩn 1.0 để RNN ổn định."),
        code('''
for run in RUNS:
    cell, mode = run.split(":")
    set_seed(SEED)
    data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
    model = RNNClassifier(cell, mode, HIDDEN, LAYERS, RNN_DROPOUT)
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, seed=SEED,
                      extra={"exercise": "E3", "cell": cell, "seq_mode": mode, "seq_len": model.seq_len,
                             "input_size": model.input_size, "hidden": HIDDEN, "layers": LAYERS,
                             "dropout": RNN_DROPOUT, "batch_size": BATCH_SIZE})
    fit(model, data, cfg, device, os.path.join(RESULTS, f"{cell}_{mode}"), name=f"E3-{cell}-{mode}")
'''),
        md("## 5. Kết quả"),
        code('''
runs, labels, colors, styles = {}, {}, {}, {}
for run in RUNS:
    cell, mode = run.split(":")
    k = (cell, mode)
    runs[k] = load_run(os.path.join(RESULTS, f"{cell}_{mode}"))
    labels[k] = f"{CELL_LABEL[cell]} · {SEQ_LABEL[mode]}"
    colors[k] = SERIES[0] if cell == "lstm" else SERIES[1]
    styles[k] = "-" if mode == "rows" else "--"
for j, b in enumerate(["mlp", "cnn"]):
    runs[b] = load_run(os.path.join(E1_RESULTS, b))
    labels[b] = f"{b.upper()} (E1)"
    colors[b] = SERIES[2 + j]
    styles[b] = "-."

table = pd.DataFrame([summary_row(labels[k], r) for k, r in runs.items()]).set_index("Mô hình")
display(table)
table.to_csv(os.path.join(FIGURES, "e3_table.csv"))
plot_curves(runs, labels, colors, styles, save_as=os.path.join(FIGURES, "e3_curves.png"))
''', analysis=True),
        md('''
## 6. Nhìn một phần ảnh: khả năng tích lũy thông tin theo thời gian

- **LSTM/GRU:** RNN một chiều nên output ở bước $t$ chỉ phụ thuộc $t$ phần đầu. Đưa output đó qua tầng phân loại
  cho biết mô hình dự đoán thế nào khi mới "đọc" được một phần ảnh — không cần huấn luyện lại.
- **MLP/CNN:** giữ $k$ hàng đầu, thay các hàng còn lại bằng pixel đen.

Không mô hình nào được huấn luyện với ảnh bị che, nên đây là phép thử như nhau cho cả hai nhóm.
'''),
        code('''
@torch.no_grad()
def rnn_prefix_accuracy(model, loader):
    model.eval()
    correct, n = None, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.logits_every_step(x)                       # (B, T, 10)
        hit = (logits.argmax(-1) == y[:, None]).sum(0).float()    # (T,)
        correct = hit if correct is None else correct + hit
        n += y.size(0)
    return (correct / n).cpu().tolist()


@torch.no_grad()
def masked_rows_accuracy(model, loader, fill, H=28):
    model.eval()
    correct, n = torch.zeros(H, device=device), 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for k in range(1, H + 1):
            xm = x.clone()
            xm[:, :, k:, :] = fill                                # chỉ giữ k hàng đầu
            correct[k - 1] += (model(xm).argmax(1) == y).sum()
        n += y.size(0)
    return (correct / n).cpu().tolist()


data = get_data(DATA_ROOT, batch_size=BATCH_SIZE, loader_seed=SEED)
prefix = {}
for run in RUNS:
    cell, mode = run.split(":")
    model = RNNClassifier(cell, mode, HIDDEN, LAYERS, RNN_DROPOUT).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS, f"{cell}_{mode}", "best.pt"), map_location=device))
    acc = rnn_prefix_accuracy(model, data.test)
    prefix[f"{cell}_{mode}"] = {"kind": "rnn", "fraction_seen": [(t + 1) / len(acc) for t in range(len(acc))],
                                "acc": acc}
for name in ["mlp", "cnn"]:
    model = build_model(name).to(device)
    model.load_state_dict(torch.load(os.path.join(E1_RESULTS, name, "best.pt"), map_location=device))
    acc = masked_rows_accuracy(model, data.test, black_value(data.mean, data.std))
    prefix[f"{name}_masked_rows"] = {"kind": "baseline", "fraction_seen": [(k + 1) / 28 for k in range(28)],
                                     "acc": acc}
save_json(prefix, os.path.join(RESULTS, "prefix_accuracy.json"))
'''),
        code('''
with open(os.path.join(RESULTS, "prefix_accuracy.json"), encoding="utf-8") as f:
    prefix = json.load(f)
fig, ax = plt.subplots(figsize=(7, 4.2))
summary = []
for name, d in prefix.items():
    if d["kind"] == "rnn":
        cell, mode = name.split("_", 1)
        c, ls = (SERIES[0] if cell == "lstm" else SERIES[1]), ("-" if mode == "rows" else "--")
        lab = f"{CELL_LABEL[cell]} · {SEQ_LABEL[mode]} (dừng ở bước t)"
    else:
        b = name.split("_")[0]
        c, ls, lab = (SERIES[2] if b == "mlp" else SERIES[3]), "-.", f"{b.upper()} (E1, che các hàng chưa thấy)"
    x, acc = 100 * np.array(d["fraction_seen"]), 100 * np.array(d["acc"])
    ax.plot(x, acc, color=c, ls=ls, label=lab)
    n = len(acc)
    summary.append({"Mô hình": lab, "25%": round(acc[n // 4 - 1], 1), "50%": round(acc[n // 2 - 1], 1),
                    "75%": round(acc[3 * n // 4 - 1], 1), "100%": round(acc[-1], 1)})
ax.axhline(10, color=MUTED, lw=0.8, ls=":")
ax.set(xlabel="phần trăm số pixel đã thấy (từ trên xuống)", ylabel="test accuracy (%)",
       title="Dự đoán khi mới thấy một phần ảnh", xlim=(0, 100), ylim=(0, 100))
ax.legend(fontsize=7, loc="upper left")
fig.savefig(os.path.join(FIGURES, "e3_prefix.png"))
plt.show()
display(pd.DataFrame(summary).set_index("Mô hình"))
''', analysis=True),
        md('''
## Ghi chú khi viết nhận xét

- LSTM vs GRU: accuracy, số tham số, thời gian; cách biểu diễn nào tốt hơn và vì sao.
- So với MLP/CNN: RNN đứng ở đâu; cấu trúc 1 chiều theo thời gian khác cấu trúc 2 chiều của CNN thế nào.
- Nhìn một phần ảnh: accuracy tăng thế nào theo số hàng đã đọc; RNN có dự đoán sớm tốt hơn không, vì sao.

## Lưu kết quả
'''),
        code(ZIP_CELL),
    ]
    write_nb("E3/E3.ipynb", cells)


if __name__ == "__main__":
    build_e1()
    build_e2()
    build_e3()
