"""Kiểm tra tính đúng của code trước khi huấn luyện (chạy trên CPU, vài giây, không cần dữ liệu).

    python -m tests.test_all          # tự chạy mọi hàm test_*
    pytest tests/test_all.py          # hoặc dùng pytest nếu đã cài
"""
import sys
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.data import split_indices
from common.engine import TrainConfig, make_optimizer, make_scheduler, train_one_epoch
from E1.models import build_model
from E2.model import ImageTransformer, copy_weights_torch_to_scratch
from E2.msa import MultiHeadSelfAttentionScratch
from E2.tokenizers import PatchTokenizer, build_tokenizer, patchify
from E3.models import RNNClassifier, rnn_param_formula, to_sequence


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- dữ liệu
def test_split_disjoint_and_complete():
    tr, va = split_indices(60000, 6000, seed=42)
    assert len(tr) == 54000 and len(va) == 6000
    assert len(set(tr.tolist()) & set(va.tolist())) == 0
    assert sorted(tr.tolist() + va.tolist()) == list(range(60000))
    tr2, _ = split_indices(60000, 6000, seed=42)
    assert torch.equal(tr, tr2), "cùng seed phải cho cùng split"


# ---------------------------------------------------------------- E1
def test_e1_shapes_and_params():
    x = torch.randn(4, 1, 28, 28)
    expected = {"softmax": 784 * 10 + 10, "mlp": 535_818, "cnn": 468_010}
    for name, n in expected.items():
        m = build_model(name)
        assert m(x).shape == (4, 10), name
        assert n_params(m) == n, f"{name}: {n_params(m)} != {n}"


# ---------------------------------------------------------------- E2: MSA
def _pair(D, h, dtype=torch.float64):
    ref = nn.MultiheadAttention(D, h, batch_first=True).to(dtype)
    mine = MultiHeadSelfAttentionScratch(D, h).to(dtype)
    mine.load_from_torch(ref)
    return ref.eval(), mine.eval()


def test_msa_output_attention_and_grad_match_torch():
    torch.manual_seed(0)
    for D, h, N in [(32, 4, 10), (96, 4, 50), (48, 6, 7), (16, 1, 5)]:
        ref, mine = _pair(D, h)
        x1 = torch.randn(3, N, D, dtype=torch.float64, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        o1, a1 = ref(x1, x1, x1, need_weights=True, average_attn_weights=False)
        o2, a2 = mine(x2, return_attn=True)
        assert o2.shape == (3, N, D) and a2.shape == (3, h, N, N)
        assert torch.allclose(o1, o2, atol=1e-10), f"output lệch D={D} h={h}"
        assert torch.allclose(a1, a2, atol=1e-10), f"attention lệch D={D} h={h}"
        g = torch.randn_like(o1)
        o1.backward(g)
        o2.backward(g)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-10), f"gradient lệch D={D} h={h}"
        assert torch.allclose(a2.sum(-1), torch.ones(3, h, N, dtype=torch.float64))


def test_msa_param_count():
    for D, h in [(32, 4), (96, 4), (64, 8)]:
        ref, mine = _pair(D, h, torch.float32)
        assert n_params(mine) == n_params(ref) == 4 * (D * D + D)


def test_msa_scratch_init_matches_torch_scheme():
    # Bias khởi tạo bằng 0 và thang đo trọng số Q/K/V cùng phân phối xavier trên ma trận (3D, D).
    torch.manual_seed(0)
    D = 96
    mine = MultiHeadSelfAttentionScratch(D, 4)
    bound = (6.0 / (D + 3 * D)) ** 0.5
    for lin in (mine.q_proj, mine.k_proj, mine.v_proj):
        assert lin.weight.abs().max() <= bound + 1e-6
        assert torch.count_nonzero(lin.bias) == 0
    assert torch.count_nonzero(mine.out_proj.bias) == 0


# ---------------------------------------------------------------- E2: tokenizer + model
def test_tokenizers_shapes():
    x = torch.randn(2, 1, 28, 28)
    expected = {"patch4": 49, "patch7": 16, "patch4s2": 169, "row": 28, "cnn": 49}
    for name, n in expected.items():
        tok = build_tokenizer(name, 64)
        assert tok.num_tokens == n, name
        assert tok(x).shape == (2, n, 64), name


def test_patch_conv_equals_patchify_plus_linear():
    torch.manual_seed(0)
    p, D = 4, 32
    tok = PatchTokenizer(D, patch_size=p)
    x = torch.randn(2, 1, 28, 28)
    lin = nn.Linear(p * p, D)
    with torch.no_grad():
        lin.weight.copy_(tok.proj.weight.reshape(D, -1))
        lin.bias.copy_(tok.proj.bias)
    assert torch.allclose(tok(x), lin(patchify(x, p)), atol=1e-5)


def test_transformer_scratch_equals_torch_after_copy():
    torch.manual_seed(0)
    x = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    for tok in ["patch4", "row", "cnn"]:
        kw = dict(tokenizer=tok, embed_dim=32, depth=2, num_heads=4)
        ref = ImageTransformer(attn_impl="torch", **kw).double().eval()
        mine = ImageTransformer(attn_impl="scratch", **kw).double().eval()
        copy_weights_torch_to_scratch(mine, ref)
        assert n_params(mine) == n_params(ref), tok
        out_ref, out_mine = ref(x), mine(x)
        assert out_mine.shape == (3, 10)
        assert torch.allclose(out_ref, out_mine, atol=1e-9), f"{tok}: lệch {(out_ref - out_mine).abs().max()}"


# ---------------------------------------------------------------- E3
def test_sequences():
    x = torch.randn(2, 1, 28, 28)
    assert torch.equal(to_sequence(x, "rows"), x[:, 0])
    assert torch.equal(to_sequence(x, "cols"), x[:, 0].transpose(1, 2))
    s = to_sequence(x, "patch7")
    assert s.shape == (2, 16, 49)
    assert torch.equal(s[:, 0], x[:, 0, :7, :7].reshape(2, 49))      # patch góc trên trái
    assert torch.equal(s[:, 1], x[:, 0, :7, 7:14].reshape(2, 49))    # patch kế bên phải
    assert torch.equal(s[:, 4], x[:, 0, 7:14, :7].reshape(2, 49))    # đầu hàng patch thứ hai
    assert to_sequence(x, "patch4").shape == (2, 49, 16)


def test_rnn_shapes_params_and_steps():
    x = torch.randn(3, 1, 28, 28)
    for cell in ["lstm", "gru"]:
        for mode, (T, F) in {"rows": (28, 28), "patch7": (16, 49)}.items():
            m = RNNClassifier(cell, mode, hidden=64, num_layers=2).eval()
            assert m(x).shape == (3, 10)
            assert n_params(m.rnn) == rnn_param_formula(cell, F, 64, 2), (cell, mode)
            steps = m.logits_every_step(x)
            assert steps.shape == (3, T, 10)
            assert torch.allclose(steps[:, -1], m(x), atol=1e-6)
    # GRU có 3 cổng, LSTM 4 cổng -> GRU = 3/4 số tham số của LSTM
    assert rnn_param_formula("gru", 28, 128, 2) * 4 == rnn_param_formula("lstm", 28, 128, 2) * 3


# ---------------------------------------------------------------- vòng lặp huấn luyện
def test_training_loop_reduces_loss():
    """Bài toán đồ chơi: 10 ảnh mẫu ngẫu nhiên, mỗi lớp là ảnh mẫu + nhiễu nhỏ.
    Mọi loại mô hình phải giảm loss rõ rệt nếu forward/backward/step được nối đúng."""
    torch.manual_seed(0)
    protos = torch.randn(10, 1, 28, 28)
    y = torch.arange(10).repeat(8)                      # 80 mẫu, 8 mẫu mỗi lớp
    x = protos[y] + 0.3 * torch.randn(80, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    models = {
        "softmax": build_model("softmax"),
        "cnn": build_model("cnn"),
        "vit-scratch": ImageTransformer("patch7", "scratch", embed_dim=32, depth=1, num_heads=2, dropout=0.0),
        "gru-rows": RNNClassifier("gru", "rows", hidden=32, num_layers=1, dropout=0.0),
    }
    crit = nn.CrossEntropyLoss()
    for name, m in models.items():
        cfg = TrainConfig(epochs=30, lr=3e-3, weight_decay=0.0)
        opt = make_optimizer(m, cfg)
        sch = make_scheduler(opt, cfg, len(loader))
        first, _ = train_one_epoch(m, loader, opt, sch, crit, torch.device("cpu"))
        for _ in range(cfg.epochs - 1):
            last, acc = train_one_epoch(m, loader, opt, sch, crit, torch.device("cpu"))
        assert last < first * 0.5, f"{name}: loss {first:.3f} -> {last:.3f}"


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} test đạt")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
