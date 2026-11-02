# CO5085 — Bài tập E1–E3

Học sâu và ứng dụng trong thị giác máy tính, HK261 (2026–2027).  
Học viên thực hiện: **Trần Hoài Tâm** — MSHV 2470743.  
Tập dữ liệu dùng chung: **Fashion-MNIST** (ảnh xám 28×28, 10 lớp).  

| Bài | Nội dung | Test acc tốt nhất | Hạn nộp |
|---|---|---|---|
| E1 | Softmax, MLP, CNN | CNN 93.51% | 02/11/2026 |
| E2 | Multi-Head Self-Attention tự viết + PyTorch, 4 cách tokenize (patch 4×4, patch 7×7, theo hàng, CNN-stem) | CNN-stem 92.07% | 02/11/2026 |
| E3 | LSTM / GRU trên chuỗi ảnh (theo hàng, theo patch), so với MLP/CNN | GRU theo hàng 90.60% | 09/11/2026 |

- Báo cáo chung: [`reports/exercises.pdf`](reports/exercises.pdf)
- Mã nguồn: <https://github.com/TamTran9x/2470743-CO5085-Exercise1>

## Môi trường

- Python ≥ 3.10, PyTorch ≥ 2.1, torchvision.
- Các package: numpy, pandas, matplotlib.

```bash
pip install -r requirements.txt
```

| Notebook | Nội dung | Thời gian (T4) |
|---|---|---|
| [`E1/E1.ipynb`](E1/E1.ipynb) | Softmax, MLP, CNN; bảng, đường cong, phân tích lỗi | ~3 phút |
| [`E2/E2.ipynb`](E2/E2.ipynb) | MSA tự viết + PyTorch, kiểm chứng, benchmark, 4 tokenizer, attention map | ~15–20 phút |
| [`E3/E3.ipynb`](E3/E3.ipynb) | LSTM/GRU trên chuỗi hàng và patch, baseline MLP/CNN, thí nghiệm nhìn một phần ảnh | ~5 phút |

Mỗi notebook **tự chứa toàn bộ code** của bài đó (không import từ repo), có thể upload lên Google Colab là chạy được từ đầu
đến cuối. Notebook được sinh từ các module bằng `python scripts/build_notebooks.py`, nên code giống hệt script.
`run_all.ipynb` chạy tất cả bằng các script trong repo.

## Run

Tất cả lệnh chạy từ thư mục gốc của repo. Dữ liệu tự tải về `data/` ở lần đầu.

```bash
python -m tests.test_all          # kiểm tra tính đúng (vài giây, CPU)

python -m E1.train --model all    # softmax, mlp, cnn            -> results/E1/<model>/

python -m E2.check_msa            # MSA tự viết vs nn.MultiheadAttention -> results/E2/msa_check.json
python -m E2.bench_msa            # thời gian MSA theo số token          -> results/E2/msa_bench.json
python -m E2.train                # 6 lần chạy tokenizer:impl            -> results/E2/<tok>_<impl>/

python -m E3.train                # lstm/gru x rows/patch7               -> results/E3/<cell>_<seq>/
python -m E3.temporal             # accuracy khi mới thấy một phần ảnh   -> results/E3/prefix_accuracy.json

python scripts/make_report_assets.py   # bảng, hình, số liệu -> reports/generated/
cd reports && xelatex exercises.tex && xelatex exercises.tex
```

Hoặc chạy tất cả: `bash run_all.sh`. Chạy thử nhanh 1 epoch mỗi mô hình: `QUICK=1 bash run_all.sh` (ghi vào `results_quick/`).

Mỗi lần huấn luyện ghi ra:

| File | Nội dung |
|---|---|
| `history.csv` | loss/accuracy train và val, learning rate, thời gian từng epoch |
| `summary.json` | số tham số, val/test accuracy, thời gian, cấu hình, môi trường |
| `best.pt` | trọng số có val accuracy cao nhất |
| `test_outputs.npz` | logits và nhãn trên tập test (để vẽ confusion matrix, tìm mẫu sai) |

Thư mục `results/` trong repo là kết quả đã dùng cho báo cáo (chạy trên Google Colab, GPU Tesla T4, PyTorch 2.11).
Checkpoint `best.pt` không được đưa lên để repo nhẹ; chạy lại lệnh huấn luyện sẽ tạo lại đúng các số liệu này
vì seed, split và thứ tự batch đã được cố định.

Một số tùy chọn hay dùng:

```bash
python -m E1.train --model cnn --epochs 20
python -m E2.train --runs patch4:scratch patch4:torch --epochs 10
python -m E2.train --runs patch4s2:scratch          # patch chồng nhau, 169 token (chậm hơn)
python -m E3.train --runs lstm:cols gru:patch4 --hidden 256
```

## Thiết lập chung

- Split cố định (seed 42): 54 000 train / 6 000 val từ tập train gốc; 10 000 test gốc chỉ dùng cho kết quả cuối.
- Chuẩn hóa bằng mean/std tính trên phần train. Không tăng cường dữ liệu.
- Vòng lặp huấn luyện tự viết (`common/engine.py`): forward → loss → `backward()` → `optimizer.step()`;
  AdamW, LR cosine theo từng bước, chọn checkpoint theo val accuracy.
- Seed, thứ tự batch và cuDNN deterministic được cố định để tái lập.

## Cấu trúc

```
common/    data.py (split, chuẩn hóa)  engine.py (training loop)  utils.py
E1/        models.py (Softmax, MLP, CNN)  train.py  E1.ipynb
E2/        msa.py (MSA tự viết + bản PyTorch)  tokenizers.py (patch, row, CNN-stem)
           model.py (Transformer encoder nông)  train.py  check_msa.py  bench_msa.py  E2.ipynb
E3/        models.py (LSTM/GRU, ảnh -> chuỗi)  train.py  temporal.py  E3.ipynb
tests/     test_all.py
scripts/   make_report_assets.py  build_notebooks.py
results/   E1/ E2/ E3/  (history.csv, summary.json, test_outputs.npz)
reports/   exercises.tex  exercises.pdf  generated/
index.html                 trang GitHub Pages
run_all.ipynb  run_all.sh
```

## GitHub Pages

Settings → Pages → Build and deployment → Source: *Deploy from a branch*, Branch: `main`, thư mục `/ (root)`.
Trang `index.html` ở gốc repo liên kết tới repo và tới `reports/exercises.pdf`.

## Sử dụng công cụ AI

Mã nguồn được viết với sự hỗ trợ của Claude (Anthropic) và ChatGPT (OpenAI), theo sự cho phép của giảng viên. Chi tiết được
khai báo trong mục "Khai báo sử dụng công cụ AI" của báo cáo.
