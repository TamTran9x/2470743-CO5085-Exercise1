#!/usr/bin/env python3
"""Sinh bảng (.tex), hình (.pdf) và macro số liệu cho báo cáo từ thư mục results/.

Chỉ cần numpy, pandas, matplotlib (không cần PyTorch), nên chạy được ở bất kỳ máy nào
đã có thư mục results/:

    python scripts/make_report_assets.py --results results --data-root data --out reports/generated

Mọi file sinh ra nằm trong reports/generated/ và được reports/exercises.tex \\input vào.
Thiếu kết quả của bài nào thì bài đó nhận bảng/hình "chưa có kết quả" thay vì báo lỗi.
"""
import argparse
import glob
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

# Bảng màu phân loại (thứ tự cố định, đã kiểm tra khả năng phân biệt cho người mù màu).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

E1_LABEL = {"softmax": "Softmax", "mlp": "MLP", "cnn": "CNN"}
TOK_LABEL = {"patch4": "Patch 4×4", "patch7": "Patch 7×7", "patch4s2": "Patch 4×4 chồng (stride 2)",
             "row": "Theo hàng", "cnn": "CNN-stem"}
IMPL_LABEL = {"scratch": "tự viết", "torch": "PyTorch"}
SEQ_LABEL = {"rows": "theo hàng", "cols": "theo cột", "patch7": "patch 7×7", "patch4": "patch 4×4"}
CELL_LABEL = {"lstm": "LSTM", "gru": "GRU"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.8,
    "savefig.bbox": "tight", "savefig.dpi": 200, "text.color": INK, "axes.titlecolor": INK,
})


# ------------------------------------------------------------------ tiện ích
class Writer:
    def __init__(self, out_dir):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.macros = {}

    def tex(self, name, text):
        with open(os.path.join(self.out, name), "w", encoding="utf-8") as f:
            f.write(text)

    def fig(self, fig, name):
        fig.savefig(os.path.join(self.out, name))
        plt.close(fig)

    def placeholder_fig(self, name, msg="Chưa có kết quả — hãy chạy run_all.ipynb"):
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2)
        self.fig(fig, name)

    def placeholder_table(self, name, msg="Chưa có kết quả — hãy chạy run\\_all.ipynb."):
        self.tex(name, "\\begin{tabular}{c}\\textit{" + msg + "}\\end{tabular}\n")

    def m(self, key, value):
        self.macros[key] = value

    def write_macros(self):
        lines = ["% Sinh tự động bởi scripts/make_report_assets.py — không sửa tay.",
                 "% Dùng trong báo cáo: \\res{E1-cnn-test} -> giá trị; thiếu khóa -> ??"]
        for k in sorted(self.macros):
            lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{self.macros[k]}}}")
        self.tex("numbers.tex", "\n".join(lines) + "\n")


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def num(n):
    return f"{int(n):,}".replace(",", "\\,")


def load_run(run_dir):
    s_path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(s_path):
        return None
    with open(s_path, encoding="utf-8") as f:
        summary = json.load(f)
    run = {"dir": run_dir, "summary": summary,
           "history": pd.read_csv(os.path.join(run_dir, "history.csv"))}
    npz = os.path.join(run_dir, "test_outputs.npz")
    if os.path.isfile(npz):
        d = np.load(npz)
        run["logits"], run["targets"] = d["logits"], d["targets"]
        run["preds"] = run["logits"].argmax(1)
    return run


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def confusion(targets, preds, k=10):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    return cm


def load_test_images(data_root):
    """Đọc ảnh test Fashion-MNIST ở định dạng IDX (thư mục torchvision tải về)."""
    base = os.path.join(data_root, "FashionMNIST", "raw", "t10k-images-idx3-ubyte")
    for path, opener in [(base, open), (base + ".gz", gzip.open)]:
        if os.path.isfile(path):
            with opener(path, "rb") as f:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    return None


def table(header, rows, align, caption_note=None):
    out = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    if caption_note:  # bọc trong một tabular ngoài để ghi chú nằm dưới bảng mà vẫn là một hộp
        out = ["\\begin{tabular}{@{}l@{}}"] + out + ["\\\\[2pt]", "{\\footnotesize " + caption_note + "}",
                                                   "\\end{tabular}"]
    return "\n".join(out) + "\n"


def bold_max(values, fmt):
    """Định dạng một cột số, in đậm giá trị lớn nhất."""
    best = max(values)
    return [("\\textbf{" + fmt(v) + "}") if v == best else fmt(v) for v in values]


def hist_macros(w, prefix, run):
    """Macro từ history: epoch có val loss thấp nhất, val loss cuối, train loss cuối, epoch tốt nhất."""
    h = run["history"]
    i = int(h["val_loss"].idxmin())
    w.m(f"{prefix}-vallossmin-epoch", str(int(h["epoch"][i])))
    w.m(f"{prefix}-vallossmin", f"{h['val_loss'][i]:.4f}")
    w.m(f"{prefix}-valloss-last", f"{h['val_loss'].iloc[-1]:.4f}")
    w.m(f"{prefix}-trainloss-last", f"{h['train_loss'].iloc[-1]:.4f}")
    w.m(f"{prefix}-bestepoch", str(run["summary"]["best_epoch"]))
    w.m(f"{prefix}-gap", f"{100 * (run['summary']['final_train_acc'] - run['summary']['best_val_acc']):.2f}")
    w.m(f"{prefix}-inftime", f"{run['summary']['test_inference_time']:.3f}")


def plot_curves(runs, labels, colors, styles, name, w, title_suffix=""):
    """2 panel: loss (train nét đứt, val nét liền) và val accuracy theo epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, run in runs.items():
        h = run["history"]
        c, ls = colors[key], styles.get(key, "-")
        ax1.plot(h["epoch"], h["train_loss"], color=c, ls=":", lw=1.4, alpha=0.9)
        ax1.plot(h["epoch"], h["val_loss"], color=c, ls=ls)
        ax2.plot(h["epoch"], 100 * h["train_acc"], color=c, ls=":", lw=1.4, alpha=0.9)
        ax2.plot(h["epoch"], 100 * h["val_acc"], color=c, ls=ls, label=labels[key])
    ax1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss" + title_suffix)
    ax2.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy" + title_suffix)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labs = ax2.get_legend_handles_labels()
    handles += [Line2D([], [], color=INK2, ls=":", lw=1.4), Line2D([], [], color=INK2, ls="-")]
    labs += ["train (chấm)", "validation (liền)"]
    fig.legend(handles, labs, loc="lower center", ncol=min(len(labs), 5), bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    w.fig(fig, name)


# ------------------------------------------------------------------ E1
def make_e1(results, data_root, w):
    runs = {k: load_run(os.path.join(results, "E1", k)) for k in E1_LABEL}
    runs = {k: v for k, v in runs.items() if v is not None}
    names = ["e1_table.tex", "e1_perclass.tex", "e1_confused_pairs.tex"]
    figs = ["e1_curves.pdf", "e1_confusion_cnn.pdf", "e1_confusion_all.pdf", "e1_errors.pdf"]
    if not runs:
        for n in names:
            w.placeholder_table(n)
        for f in figs:
            w.placeholder_fig(f)
        return runs

    keys = list(runs)
    S = {k: runs[k]["summary"] for k in keys}
    test_accs = [S[k]["test_acc"] for k in keys]
    test_fmt = dict(zip(keys, bold_max(test_accs, pct), strict=True))
    rows = []
    for k in keys:
        s = S[k]
        rows.append([E1_LABEL[k], num(s["params"]), pct(s["final_train_acc"]), pct(s["best_val_acc"]),
                     s["best_epoch"], test_fmt[k], f"{s['test_loss']:.4f}", f"{s['mean_epoch_time']:.1f}"])
        p = f"E1-{k}"
        w.m(f"{p}-params", num(s["params"]))
        w.m(f"{p}-test", pct(s["test_acc"]))
        w.m(f"{p}-val", pct(s["best_val_acc"]))
        w.m(f"{p}-train", pct(s["final_train_acc"]))
        w.m(f"{p}-gap", f"{100 * (s['final_train_acc'] - s['best_val_acc']):.2f}")
        w.m(f"{p}-epochtime", f"{s['mean_epoch_time']:.1f}")
        w.m(f"{p}-bestepoch", str(s["best_epoch"]))
        w.m(f"{p}-epochs", str(s["config"]["epochs"]))
        hist_macros(w, p, runs[k])
        if "preds" in runs[k]:
            w.m(f"{p}-errors", num(int((runs[k]["preds"] != runs[k]["targets"]).sum())))
    first = S[keys[0]]
    w.m("data-train", num(first["data_sizes"]["train"]))
    w.m("data-val", num(first["data_sizes"]["val"]))
    w.m("data-test", num(first["data_sizes"]["test"]))
    w.m("data-mean", f"{first['norm_mean']:.4f}")
    w.m("data-std", f"{first['norm_std']:.4f}")
    w.m("env-device", first["env"].get("device_name", "?").replace("_", "\\_"))
    w.m("env-torch", first["env"].get("torch", "?"))
    w.tex("e1_table.tex", table(
        ["Mô hình", "Số tham số", "Train acc (\\%)", "Val acc tốt nhất (\\%)", "Epoch tốt nhất",
         "Test acc (\\%)", "Test loss", "Giây/epoch"], rows, "lrrrrrrr",
        "Train acc lấy ở epoch cuối (có dropout); test dùng checkpoint có val acc cao nhất."))

    colors = {k: SERIES[i] for i, k in enumerate(E1_LABEL) if k in runs}
    plot_curves(runs, E1_LABEL, colors, {}, "e1_curves.pdf", w)

    have_preds = [k for k in keys if "preds" in runs[k]]
    if not have_preds:
        for n in ["e1_perclass.tex", "e1_confused_pairs.tex"]:
            w.placeholder_table(n)
        for f in ["e1_confusion_cnn.pdf", "e1_confusion_all.pdf", "e1_errors.pdf"]:
            w.placeholder_fig(f)
        return runs

    cms = {k: confusion(runs[k]["targets"], runs[k]["preds"]) for k in have_preds}

    # Accuracy theo từng lớp
    rows = []
    for c in range(10):
        vals = [cms[k][c, c] / cms[k][c].sum() for k in have_preds]
        rows.append([CLASSES[c]] + bold_max(vals, pct))
    w.tex("e1_perclass.tex", table(["Lớp"] + [E1_LABEL[k] for k in have_preds], rows,
                                   "l" + "r" * len(have_preds)))
    for k in have_preds:
        accs = np.diag(cms[k]) / cms[k].sum(1)
        w.m(f"E1-{k}-worstclass", CLASSES[int(accs.argmin())])
        w.m(f"E1-{k}-worstclassacc", pct(accs.min()))

    # Các cặp lớp hay nhầm nhất (theo mô hình tốt nhất có kết quả, ưu tiên CNN)
    ref = "cnn" if "cnn" in cms else have_preds[-1]
    off = cms[ref].copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for idx in np.argsort(off, axis=None)[::-1][:6]:
        t, p = np.unravel_index(idx, off.shape)
        pairs.append((t, p))
    rows = [[CLASSES[t], CLASSES[p]] + [cms[k][t, p] for k in have_preds] for t, p in pairs]
    w.tex("e1_confused_pairs.tex", table(
        ["Nhãn thật", "Bị đoán là"] + [E1_LABEL[k] for k in have_preds], rows,
        "ll" + "r" * len(have_preds), f"Sắp theo số lỗi của {E1_LABEL[ref]} trên 10\\,000 ảnh test."))
    t, p = pairs[0]
    w.m("E1-toppair-true", CLASSES[t])
    w.m("E1-toppair-pred", CLASSES[p])
    w.m("E1-toppair-count", str(cms[ref][t, p]))

    # Tỉ lệ lỗi nằm trong nhóm áo phần thân trên (T-shirt/top, Pullover, Coat, Shirt)
    upper = [0, 2, 4, 6]
    for k in have_preds:
        cm_k = cms[k]
        total_err = cm_k.sum() - np.trace(cm_k)
        inside = sum(cm_k[a, b] for a in upper for b in upper if a != b)
        if total_err:
            w.m(f"E1-{k}-upperbody-share", pct(inside / total_err, 1))

    # Mức độ chồng lấp lỗi giữa các mô hình
    if {"mlp", "cnn"} <= set(have_preds):
        err_cnn = runs["cnn"]["preds"] != runs["cnn"]["targets"]
        err_mlp = runs["mlp"]["preds"] != runs["mlp"]["targets"]
        if err_cnn.sum() > 0:
            w.m("E1-cnnerr-also-mlp", pct((err_cnn & err_mlp).sum() / err_cnn.sum(), 1))
        if "softmax" in have_preds:
            err_sm = runs["softmax"]["preds"] != runs["softmax"]["targets"]
            w.m("E1-allwrong", num(int((err_cnn & err_mlp & err_sm).sum())))

    # Confusion matrix của CNN (số lượng)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cm = cms[ref]
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="lớp dự đoán", ylabel="lớp thật", title=f"Confusion matrix — {E1_LABEL[ref]} (tập test)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="tỉ lệ theo hàng")
    w.fig(fig, "e1_confusion_cnn.pdf")

    # Ba confusion matrix (chuẩn hóa theo hàng, bỏ đường chéo để thấy rõ lỗi)
    fig, axes = plt.subplots(1, len(have_preds), figsize=(3.6 * len(have_preds), 3.6), squeeze=False)
    vmax = max((cms[k] / cms[k].sum(1, keepdims=True) * (1 - np.eye(10))).max() for k in have_preds)
    for ax, k in zip(axes[0], have_preds, strict=True):
        e = cms[k] / cms[k].sum(1, keepdims=True) * (1 - np.eye(10))
        im = ax.imshow(e, cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_xticks(range(10), SHORT, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(10), SHORT, fontsize=7)
        ax.set_title(f"{E1_LABEL[k]} — test acc {pct(S[k]['test_acc'])}%")
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02, label="tỉ lệ lỗi (ngoài đường chéo)")
    w.fig(fig, "e1_confusion_all.pdf")

    # Các mẫu bị đoán sai tự tin nhất
    images = load_test_images(data_root)
    if images is None:
        w.placeholder_fig("e1_errors.pdf", "Không tìm thấy ảnh test trong data/ — cần chạy E1 trước để tải dữ liệu")
    else:
        prob = softmax(runs[ref]["logits"])
        tg, pr = runs[ref]["targets"], runs[ref]["preds"]
        wrong = np.where(pr != tg)[0]
        order = wrong[np.argsort(-prob[wrong, pr[wrong]])][:16]
        fig, axes = plt.subplots(2, 8, figsize=(11, 4.6))
        for ax, i in zip(axes.flat, order, strict=False):
            ax.imshow(images[i], cmap="gray_r")
            ax.set_title(f"thật: {CLASSES[tg[i]]}\nđoán: {CLASSES[pr[i]]}\np={prob[i, pr[i]]:.2f}",
                         fontsize=6.5, color=INK)
            ax.axis("off")
        for ax in list(axes.flat)[len(order):]:
            ax.axis("off")
        fig.suptitle(f"16 ảnh test {E1_LABEL[ref]} đoán sai với độ tự tin cao nhất", fontsize=10)
        fig.tight_layout(h_pad=1.5)
        w.fig(fig, "e1_errors.pdf")
    return runs


# ------------------------------------------------------------------ E2
def make_e2(results, w):
    e2 = os.path.join(results, "E2")
    runs = {}
    for d in sorted(glob.glob(os.path.join(e2, "*", "summary.json"))):
        r = load_run(os.path.dirname(d))
        ex = r["summary"]["config"]["extra"]
        runs[(ex["tokenizer"], ex["attn_impl"])] = r

    # Kiểm chứng MSA
    chk_path = os.path.join(e2, "msa_check.json")
    if os.path.isfile(chk_path):
        with open(chk_path, encoding="utf-8") as f:
            chk = json.load(f)
        rows = [[r["embed_dim"], r["num_heads"], r["num_tokens"], r["dtype"],
                 f"{r['max_abs_diff_output']:.1e}", f"{r['max_abs_diff_attention']:.1e}",
                 f"{r['max_abs_diff_input_grad']:.1e}", "đạt" if r["passed"] else "\\textbf{lệch}"]
                for r in chk["results"]]
        w.tex("e2_msa_check.tex", table(
            ["$D$", "$h$", "$N$", "kiểu số", "lệch output", "lệch attention", "lệch gradient", "kết quả"],
            rows, "rrrlrrrl", "Sai lệch tuyệt đối lớn nhất giữa bản tự viết và nn.MultiheadAttention "
                              f"khi dùng chung trọng số (chạy trên {chk['device']})."))
        f64 = [r for r in chk["results"] if r["dtype"] == "float64"]
        f32 = [r for r in chk["results"] if r["dtype"] == "float32"]
        if f64:
            w.m("E2-check-f64", f"{max(r['max_abs_diff_output'] for r in f64):.0e}")
        if f32:
            w.m("E2-check-f32", f"{max(r['max_abs_diff_output'] for r in f32):.0e}")
        w.m("E2-check-allpass", "đạt" if all(r["passed"] for r in chk["results"]) else "KHÔNG đạt")
    else:
        w.placeholder_table("e2_msa_check.tex")

    # Benchmark theo số token
    b_path = os.path.join(e2, "msa_bench.json")
    if os.path.isfile(b_path):
        with open(b_path, encoding="utf-8") as f:
            bench = json.load(f)
        br = bench["results"]

        def fmt(v, nd=2):
            return "—" if v is None else f"{v:.{nd}f}"

        rows = [[r["num_tokens"], fmt(r.get("scratch_ms")), fmt(r.get("torch_ms")),
                 fmt(r["scratch_ms"] / r["torch_ms"]) if r.get("scratch_ms") and r.get("torch_ms") else "—",
                 fmt(r.get("scratch_peak_mem_mb"), 0), fmt(r.get("torch_peak_mem_mb"), 0),
                 fmt(r["attn_matrix_mb"], 0)] for r in br]
        w.tex("e2_bench.tex", table(
            ["$N$", "tự viết (ms)", "PyTorch (ms)", "tỉ lệ", "bộ nhớ tự viết (MB)", "bộ nhớ PyTorch (MB)",
             "ma trận $A$ (MB)"], rows, "rrrrrrr",
            f"Một lớp MSA, forward + backward, batch {br[0]['batch']}, $D={bench['embed_dim']}$, "
            f"$h={bench['heads']}$, trung bình {bench['iters']} lượt trên {bench['device']}."))
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for impl, c in [("scratch", SERIES[0]), ("torch", SERIES[1])]:
            pts = [(r["num_tokens"], r[f"{impl}_ms"]) for r in br if r.get(f"{impl}_ms")]
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, marker="o", ms=5, color=c, label=f"MSA {IMPL_LABEL[impl]}")
        ax.set(xscale="log", yscale="log", xlabel="số token N", ylabel="ms / lượt (forward + backward)",
               title="Chi phí một lớp MSA theo số token")
        ticks = []  # bỏ nhãn của các mốc quá gần nhau (vd. 169 và 196) để không chồng chữ
        for n in sorted(r["num_tokens"] for r in br):
            if ticks and n / ticks[-1] < 1.3:
                ticks[-1] = n
            else:
                ticks.append(n)
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.minorticks_off()
        ax.legend()
        w.fig(fig, "e2_bench.pdf")
        by_n = {r["num_tokens"]: r for r in br}
        for r in br:
            n = r["num_tokens"]
            for impl in ("scratch", "torch"):
                if r.get(f"{impl}_ms") is not None:
                    w.m(f"E2-bench-{n}-{impl}-ms", f"{r[f'{impl}_ms']:.1f}")
                if r.get(f"{impl}_peak_mem_mb") is not None:
                    w.m(f"E2-bench-{n}-{impl}-mem", num(round(r[f"{impl}_peak_mem_mb"])))
            if r.get("scratch_peak_mem_mb") and r.get("torch_peak_mem_mb"):
                w.m(f"E2-bench-{n}-memratio", f"{r['scratch_peak_mem_mb'] / r['torch_peak_mem_mb']:.1f}")
            w.m(f"E2-bench-{n}-attnmb", num(round(r["attn_matrix_mb"])))
        if 49 in by_n and 784 in by_n and by_n[49].get("scratch_ms") and by_n[784].get("scratch_ms"):
            w.m("E2-bench-ratio-784-49", f"{by_n[784]['scratch_ms'] / by_n[49]['scratch_ms']:.0f}")
    else:
        w.placeholder_table("e2_bench.tex")
        w.placeholder_fig("e2_bench.pdf")

    if not runs:
        for n in ["e2_table.tex", "e2_impl_compare.tex"]:
            w.placeholder_table(n)
        w.placeholder_fig("e2_curves.pdf")
        return runs

    tok_order = [t for t in TOK_LABEL if any(k[0] == t for k in runs)]
    keys = sorted(runs, key=lambda k: (tok_order.index(k[0]), k[1] != "scratch"))
    test_fmt = dict(zip(keys, bold_max([runs[k]["summary"]["test_acc"] for k in keys], pct), strict=True))
    rows = []
    for k in keys:
        s = runs[k]["summary"]
        ex = s["config"]["extra"]
        rows.append([TOK_LABEL[k[0]], IMPL_LABEL[k[1]], ex["num_tokens"], num(s["params"]),
                     pct(s["final_train_acc"]), pct(s["best_val_acc"]), test_fmt[k],
                     f"{s['mean_epoch_time']:.1f}", f"{s['test_inference_time']:.2f}"])
        p = f"E2-{k[0]}-{k[1]}"
        w.m(f"{p}-test", pct(s["test_acc"]))
        w.m(f"{p}-val", pct(s["best_val_acc"]))
        w.m(f"{p}-train", pct(s["final_train_acc"]))
        w.m(f"{p}-params", num(s["params"]))
        w.m(f"{p}-epochtime", f"{s['mean_epoch_time']:.1f}")
        w.m(f"{p}-tokens", str(ex["num_tokens"]))
        hist_macros(w, p, runs[k])
    any_s = runs[keys[0]]["summary"]
    ex = any_s["config"]["extra"]
    for key in ["embed_dim", "depth", "heads", "mlp_ratio", "dropout"]:
        w.m(f"E2-cfg-{key.replace('_', '')}", str(ex[key]))
    w.m("E2-cfg-epochs", str(any_s["config"]["epochs"]))
    w.m("E2-cfg-wd", str(any_s["config"]["weight_decay"]))
    w.tex("e2_table.tex", table(
        ["Tokenizer", "MSA", "Số token", "Số tham số", "Train acc", "Val acc", "Test acc", "Giây/epoch",
         "Suy luận test (s)"], rows, "llrrrrrrr",
        "Accuracy tính theo \\%. Số token đã gồm token CLS. Suy luận: toàn bộ 10\\,000 ảnh test."))

    rows = []
    for tok in tok_order:
        if (tok, "scratch") in runs and (tok, "torch") in runs:
            a, b = runs[(tok, "scratch")]["summary"], runs[(tok, "torch")]["summary"]
            rows.append([TOK_LABEL[tok], pct(a["test_acc"]), pct(b["test_acc"]),
                         f"{100 * (a['test_acc'] - b['test_acc']):+.2f}",
                         f"{a['mean_epoch_time']:.1f}", f"{b['mean_epoch_time']:.1f}",
                         f"{a['mean_epoch_time'] / b['mean_epoch_time']:.2f}"])
            w.m(f"E2-{tok}-timeratio", f"{a['mean_epoch_time'] / b['mean_epoch_time']:.2f}")
            w.m(f"E2-{tok}-infratio", f"{a['test_inference_time'] / b['test_inference_time']:.2f}")
            w.m(f"E2-{tok}-accdiff", f"{100 * abs(a['test_acc'] - b['test_acc']):.2f}")
            hs, ht = runs[(tok, "scratch")]["history"], runs[(tok, "torch")]["history"]
            n = min(len(hs), len(ht))
            w.m(f"E2-{tok}-maxvaldiff",
                f"{100 * np.abs(hs['val_acc'].values[:n] - ht['val_acc'].values[:n]).max():.2f}")
    if rows:
        w.tex("e2_impl_compare.tex", table(
            ["Tokenizer", "Test acc tự viết", "Test acc PyTorch", "Chênh (điểm \\%)", "Giây/epoch tự viết",
             "Giây/epoch PyTorch", "Tỉ lệ thời gian"], rows, "lrrrrrr",
            "Hai bản xuất phát từ cùng trọng số khởi tạo và cùng thứ tự batch."))
    else:
        w.placeholder_table("e2_impl_compare.tex", "Chưa có cặp tự viết/PyTorch cùng tokenizer.")

    labels = {k: f"{TOK_LABEL[k[0]]} · {IMPL_LABEL[k[1]]}" for k in keys}
    colors = {k: SERIES[tok_order.index(k[0])] for k in keys}
    styles = {k: "-" if k[1] == "scratch" else "--" for k in keys}
    plot_curves({k: runs[k] for k in keys}, labels, colors, styles, "e2_curves.pdf", w)
    return runs


# ------------------------------------------------------------------ E3
def make_e3(results, e1_runs, w):
    e3 = os.path.join(results, "E3")
    runs = {}
    for d in sorted(glob.glob(os.path.join(e3, "*", "summary.json"))):
        r = load_run(os.path.dirname(d))
        ex = r["summary"]["config"]["extra"]
        runs[(ex["cell"], ex["seq_mode"])] = r
    if not runs:
        w.placeholder_table("e3_table.tex")
        w.placeholder_fig("e3_curves.pdf")
        w.placeholder_fig("e3_prefix.pdf")
        return runs

    seq_order = [s for s in SEQ_LABEL if any(k[1] == s for k in runs)]
    keys = sorted(runs, key=lambda k: (seq_order.index(k[1]), k[0]))
    all_rows = []
    for k in keys:
        s = runs[k]["summary"]
        ex = s["config"]["extra"]
        all_rows.append((f"{CELL_LABEL[k[0]]}", f"chuỗi {SEQ_LABEL[k[1]]}",
                         f"{ex['seq_len']} × {ex['input_size']}", s))
        p = f"E3-{k[0]}-{k[1]}"
        w.m(f"{p}-test", pct(s["test_acc"]))
        w.m(f"{p}-val", pct(s["best_val_acc"]))
        w.m(f"{p}-train", pct(s["final_train_acc"]))
        w.m(f"{p}-params", num(s["params"]))
        w.m(f"{p}-epochtime", f"{s['mean_epoch_time']:.1f}")
        hist_macros(w, p, runs[k])
    for b, desc in [("mlp", "vector 784"), ("cnn", "ảnh 28 × 28")]:
        if b in e1_runs:
            all_rows.append((f"{E1_LABEL[b]} (E1)", "không hồi quy", desc, e1_runs[b]["summary"]))
    ex = runs[keys[0]]["summary"]["config"]
    w.m("E3-cfg-hidden", str(ex["extra"]["hidden"]))
    w.m("E3-cfg-layers", str(ex["extra"]["layers"]))
    w.m("E3-cfg-epochs", str(ex["epochs"]))
    w.m("E3-cfg-clip", str(ex["grad_clip"]))
    test_fmt = bold_max([r[3]["test_acc"] for r in all_rows], pct)
    rows = [[a, b, c, num(s["params"]), pct(s["final_train_acc"]), pct(s["best_val_acc"]), t,
             f"{s['mean_epoch_time']:.1f}"] for (a, b, c, s), t in zip(all_rows, test_fmt, strict=True)]
    w.tex("e3_table.tex", table(
        ["Mô hình", "Đầu vào", "$T \\times F$", "Số tham số", "Train acc", "Val acc", "Test acc", "Giây/epoch"],
        rows, "lllrrrrr", "Accuracy tính theo \\%. MLP và CNN lấy từ E1 (cùng split, cùng seed)."))

    curve_runs, labels, colors, styles = {}, {}, {}, {}
    for k in keys:
        curve_runs[k] = runs[k]
        labels[k] = f"{CELL_LABEL[k[0]]} · {SEQ_LABEL[k[1]]}"
        colors[k] = SERIES[0] if k[0] == "lstm" else SERIES[1]
        styles[k] = "-" if seq_order.index(k[1]) == 0 else "--"
    for j, b in enumerate(["mlp", "cnn"]):
        if b in e1_runs:
            curve_runs[b] = e1_runs[b]
            labels[b] = f"{E1_LABEL[b]} (E1)"
            colors[b] = SERIES[2 + j]
            styles[b] = "-."
    plot_curves(curve_runs, labels, colors, styles, "e3_curves.pdf", w)

    pa_path = os.path.join(e3, "prefix_accuracy.json")
    if os.path.isfile(pa_path):
        with open(pa_path, encoding="utf-8") as f:
            pa = json.load(f)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for name, d in pa.items():
            if d["kind"] == "rnn":
                cell, mode = name.split("_", 1)
                c = SERIES[0] if cell == "lstm" else SERIES[1]
                ls = "-" if mode == "rows" else "--"
                lab = f"{CELL_LABEL[cell]} · {SEQ_LABEL.get(mode, mode)} (dừng ở bước t)"
            else:
                b = name.split("_")[0]
                c = SERIES[2] if b == "mlp" else SERIES[3]
                ls, lab = "-.", f"{E1_LABEL[b]} (E1, che các hàng chưa thấy)"
            x = 100 * np.array(d["fraction_seen"])
            ax.plot(x, 100 * np.array(d["acc"]), color=c, ls=ls, label=lab)
            key = name.replace("_", "-")
            acc = d["acc"]
            n = len(acc)
            w.m(f"E3-prefix-{key}-half", pct(acc[n // 2 - 1], 1))
            w.m(f"E3-prefix-{key}-quarter", pct(acc[n // 4 - 1], 1))
            w.m(f"E3-prefix-{key}-quarter3", pct(acc[3 * n // 4 - 1], 1))
        ax.axhline(10, color=MUTED, lw=0.8, ls=":")
        ax.text(99, 11.5, "đoán ngẫu nhiên", color=MUTED, fontsize=7, ha="right")
        ax.set(xlabel="phần trăm số pixel mô hình đã thấy (từ trên xuống)", ylabel="test accuracy (%)",
               title="Dự đoán khi mới thấy một phần ảnh", ylim=(0, 100), xlim=(0, 100))
        ax.legend(fontsize=7, loc="upper left")
        w.fig(fig, "e3_prefix.pdf")
    else:
        w.placeholder_fig("e3_prefix.pdf")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="reports/generated")
    args = ap.parse_args()
    w = Writer(args.out)
    e1 = make_e1(args.results, args.data_root, w)
    e2 = make_e2(args.results, w)
    e3 = make_e3(args.results, e1, w)
    w.write_macros()
    print(f"E1: {len(e1)} mô hình | E2: {len(e2)} lần chạy | E3: {len(e3)} lần chạy -> {args.out}")


if __name__ == "__main__":
    main()
