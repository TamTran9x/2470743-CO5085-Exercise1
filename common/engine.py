"""Vòng lặp huấn luyện tự viết, dùng chung cho E1, E2, E3.

Mỗi bước huấn luyện đi đúng trình tự đề yêu cầu:
    forward -> loss -> backward() -> optimizer.step()
Không dùng API cấp cao kiểu trainer.fit().
"""
import csv
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .utils import Timer, count_params, environment_info, save_json


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: float = 0.0      # số epoch tăng LR tuyến tính từ 0 trước khi giảm theo cosine
    grad_clip: float = 0.0          # 0 = không cắt gradient
    label_smoothing: float = 0.0
    seed: int = 42
    extra: dict = field(default_factory=dict)  # thông tin riêng của từng bài (kiến trúc, tokenizer...)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; không áp weight decay lên bias, LayerNorm/BatchNorm, CLS token và positional embedding."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def make_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """LR theo từng bước: warmup tuyến tính rồi giảm theo cosine về 0."""
    total = cfg.epochs * steps_per_epoch
    warmup = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=0.0):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)                    # 1. forward: (B, 10) logits, CHƯA qua softmax
        loss = criterion(logits, y)          # 2. loss: CrossEntropyLoss = log-softmax + NLL

        optimizer.zero_grad(set_to_none=True)
        loss.backward()                      # 3. backward: tính gradient
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()                     # 4. cập nhật trọng số
        scheduler.step()                     # LR thay đổi theo từng bước

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_outputs=False):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
        if return_outputs:
            all_logits.append(logits.float().cpu())
            all_targets.append(y.cpu())
    if return_outputs:
        return total_loss / n, correct / n, torch.cat(all_logits), torch.cat(all_targets)
    return total_loss / n, correct / n


def fit(model: nn.Module, data, cfg: TrainConfig, device, out_dir: str, name: str) -> dict:
    """Huấn luyện, chọn checkpoint tốt nhất theo val accuracy, đánh giá trên test, lưu log.

    Ghi ra out_dir:
        history.csv   - loss/acc train & val, LR, thời gian từng epoch
        best.pt       - trọng số có val accuracy cao nhất
        test_outputs.npz - logits và nhãn trên tập test (để vẽ confusion matrix, tìm mẫu sai)
        summary.json  - tóm tắt số liệu + cấu hình + môi trường
    """
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, len(data.train))

    n_params = count_params(model)
    print(f"[{name}] params = {n_params:,} | device = {device}")

    history = []
    best_val_acc, best_epoch = -1.0, -1
    ckpt_path = os.path.join(out_dir, "best.pt")
    for epoch in range(1, cfg.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        with Timer(device) as t:
            tr_loss, tr_acc = train_one_epoch(model, data.train, optimizer, scheduler,
                                              criterion, device, cfg.grad_clip)
        va_loss, va_acc = evaluate(model, data.val, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "lr": lr_now, "epoch_time": t.elapsed,
        })
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
        print(f"[{name}] epoch {epoch:2d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {t.elapsed:.1f}s")

    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Đánh giá trên test bằng checkpoint tốt nhất theo val (không chọn theo test).
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    te_loss, te_acc, logits, targets = evaluate(model, data.test, criterion, device, return_outputs=True)
    np.savez_compressed(os.path.join(out_dir, "test_outputs.npz"),
                        logits=logits.numpy().astype(np.float32), targets=targets.numpy())

    # Thời gian suy luận trên toàn bộ tập test (sau 1 lượt chạy làm nóng).
    evaluate(model, data.test, criterion, device)
    with Timer(device) as t_inf:
        evaluate(model, data.test, criterion, device)

    epoch_times = [h["epoch_time"] for h in history]
    summary = {
        "name": name,
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "final_train_acc": history[-1]["train_acc"],
        "final_train_loss": history[-1]["train_loss"],
        "mean_epoch_time": float(np.mean(epoch_times[1:] if len(epoch_times) > 1 else epoch_times)),
        "total_train_time": float(np.sum(epoch_times)),
        "test_inference_time": t_inf.elapsed,
        "data_sizes": data.sizes,
        "norm_mean": data.mean,
        "norm_std": data.std,
        "config": asdict(cfg),
        "env": environment_info(),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    print(f"[{name}] best val acc {best_val_acc:.4f} (epoch {best_epoch}) | test acc {te_acc:.4f}")
    return summary
