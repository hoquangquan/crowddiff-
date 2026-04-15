# CrowdDiff: Đếm Đám Đông Bằng Mô Hình Khuếch Tán

> **Đồ án môn học**  
> Dataset: **ShanghaiTech Part A** · Phương pháp: **Diffusion Model** · Mục tiêu: Đếm người · Heatmap · Đánh giá sai số

<p align="center">
  <img src="figs/flow chart.jpg" width="820"/>
</p>

---

## Mục lục

1. [Giới thiệu & Mục tiêu](#1-giới-thiệu--mục-tiêu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Dataset: ShanghaiTech Part A](#3-dataset-shanghaitech-part-a)
4. [Cấu trúc Project](#4-cấu-trúc-project)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Hướng dẫn chạy Pipeline](#6-hướng-dẫn-chạy-pipeline)
   - [6.1 Chạy tự động (1 lệnh)](#61-chạy-tự-động-1-lệnh)
   - [6.2 Chạy từng bước thủ công](#62-chạy-từng-bước-thủ-công)
   - [6.3 Chạy trên Cloud (Colab & Kaggle)](#63-chạy-trên-cloud-colab--kaggle)
7. [Phân tích kết quả (Notebooks)](#7-phân-tích-kết-quả--notebooks)
8. [Kết quả & Đánh giá](#8-kết-quả--đánh-giá)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Giới thiệu & Mục tiêu

Dự án này triển khai **CrowdDiff** — một phương pháp đếm đám đông tiên tiến sử dụng **Mô hình Khuếch tán (Diffusion Model)** để sinh ra bản đồ mật độ đám đông (density map) từ ảnh tĩnh.

### Phạm vi của đồ án

| Tiêu chí | Chi tiết |
|---|---|
| **Loại đầu vào** | Ảnh tĩnh (không xử lý video) |
| **Dataset** | ShanghaiTech Part A (duy nhất) |
| **Pipeline** | 1 pipeline chính: Preprocess → Train → Inference → Evaluate |
| **Tính năng cốt lõi** | Đếm số người · Tạo Heatmap · Đánh giá MAE/MSE |

### Tại sao dùng Diffusion Model?

Các phương pháp CNN truyền thống như CSRNet hay DM-Count chỉ đưa ra **một kết quả tất định** (deterministic). CrowdDiff khai thác tính **ngẫu nhiên có kiểm soát** (stochastic) của Diffusion Model: mỗi lần sinh ra một density map khác nhau, sau đó chọn kết quả tối ưu nhất — giúp tăng độ chính xác trong ảnh đám đông dày đặc.

---

## 2. Cơ sở lý thuyết

### 2.1 Diffusion Model (Mô hình Khuếch tán)

Quá trình huấn luyện gồm 2 chiều:

- **Forward Process (Thêm nhiễu)**: Bản đồ mật độ gốc `x₀` dần bị phá hủy bởi nhiễu Gaussian qua `T` bước thời gian cho đến khi trở thành nhiễu thuần túy `xₜ`.

  ```
  q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜ·I)
  ```

- **Reverse Process (Khử nhiễu)**: Mạng U-Net học cách đảo ngược quá trình trên, từ nhiễu `xₜ` và ảnh đám đông (điều kiện) để khôi phục density map `x₀`.

  ```
  pθ(xₜ₋₁ | xₜ, image) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))
  ```

### 2.2 Multi-hypothesis Sampling (Lấy mẫu Đa giả thuyết)

Khác với hồi quy thông thường, mô hình khuếch tán cho phép lấy mẫu **N lần** từ cùng 1 ảnh. Tại mỗi lần:

1. Sinh density map từ nhiễu ngẫu nhiên
2. Tính toán số người từ density map
3. So sánh với Ground Truth → Giữ lại mẫu tốt nhất

<p float="left">
  <img src="figs/gt 361.jpg" width="200"/>
  <img src="figs/trial1 349.jpg" width="200"/>
  <img src="figs/trial2 351.jpg" width="200"/>
  <img src="figs/final 359.jpg" width="200"/>
</p>

*GT: 361 người · Trial 1: 349 · Trial 2: 351 · Final: 359*

### 2.3 Density Map & Crowd Counting

Density map là ảnh grayscale trong đó mỗi điểm sáng tương ứng với vị trí của một người. Tổng pixel của density map (sau khi chuẩn hóa) bằng đúng số người.

```
Số người = ΣΣ density_map(x, y)
```

---

## 3. Dataset: ShanghaiTech Part A

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn** | [Kaggle – tthien/shanghaitech](https://www.kaggle.com/datasets/tthien/shanghaitech) |
| **Tập train** | 300 ảnh |
| **Tập test** | 182 ảnh |
| **Số người / ảnh** | Trung bình ~501 (max ~3139) |
| **Đặc điểm** | Đám đông dày đặc, góc nhìn phức tạp, chất lượng cao |

Part A là phần khó nhất trong ShanghaiTech với đám đông cực kỳ dày đặc, lý tưởng để kiểm tra mô hình trong điều kiện thực tế khắc nghiệt.

---

## 4. Cấu trúc Project

```
crowddiff-main/
│
├── 📁 guided_diffusion/          # Kiến trúc mô hình khuếch tán (OpenAI)
│   ├── gaussian_diffusion.py     # Thuật toán forward/reverse diffusion
│   ├── unet.py                   # Mạng U-Net (backbone)
│   ├── train_util.py             # Vòng lặp huấn luyện (TrainLoop)
│   └── script_util.py            # Factory tạo model + diffusion
│
├── 📁 cc_utils/                  # Tiện ích Crowd Counting
│   ├── preprocess_shtech.py      # Tiền xử lý dataset ShanghaiTech
│   ├── utils.py                  # DataParameter, crop/combine logic
│   ├── evaluate.py               # Tính MAE/MSE + xuất heatmap overlay
│   └── vis_test.py               # Trực quan hóa kết quả
│
├── 📁 scripts/                   # Scripts chính
│   ├── super_res_train.py        # ▶ Huấn luyện mô hình
│   ├── super_res_sample.py       # ▶ Suy diễn / sinh density map
│   └── visualize_results.py      # ▶ Trực quan hoá GT, Heatmap & Peaks
│
├── 📁 sh_scripts/                # Shell scripts tiện lợi
│   ├── preprocess_shtech.sh      # Tiền xử lý ShanghaiTech
│   ├── train_diff.sh             # Bắt đầu huấn luyện
│   └── test_diff.sh              # Bắt đầu suy diễn
│
├── 📁 notebooks/                 # Jupyter Notebooks phân tích
│   ├── 01_explore_dataset.ipynb  # Khám phá dataset: thống kê, heatmap GT
│   └── 03_evaluate_results.ipynb # Đánh giá kết quả: MAE/MSE, scatter, heatmap
│
├── 📁 results/                   # Kết quả suy diễn và model checkpoints
├── 📁 weights/                   # Pretrained weights
├── 📁 DuLieuDaXuLy/              # Dữ liệu đã tiền xử lý
├── 📁 figs/                      # Hình ảnh minh họa
│
├── run_pipeline.sh               # ▶▶ Chạy toàn bộ pipeline 1 lệnh
└── requirements.txt              # Các thư viện Python cần thiết
```

---

## 5. Cài đặt môi trường

**Yêu cầu**: Python 3.9.7, PyTorch 1.13.1, CUDA (tùy chọn)

```bash
# Bước 1: Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Bước 2: Cài đặt dependencies
pip install -r requirements.txt

# Bước 3: (Tùy chọn) Cài Jupyter để chạy notebooks
pip install jupyter notebook
```

---

## 6. Hướng dẫn chạy Pipeline

### 6.1 Chạy tự động (1 lệnh)

```bash
# Chạy toàn bộ pipeline từ đầu đến cuối
bash run_pipeline.sh

# Hoặc chỉ chạy một số bước cụ thể
bash run_pipeline.sh --steps download,preprocess
bash run_pipeline.sh --steps sample,evaluate --model_path results/model000020.pt

# Xem hướng dẫn đầy đủ
bash run_pipeline.sh --help
```

---

### 6.2 Chạy từng bước thủ công

#### Bước 1 — Tải Dataset từ Kaggle

```bash
.venv/bin/python -c "
import kagglehub
path = kagglehub.dataset_download('tthien/shanghaitech')
print(f'Dataset tải về: {path}')
"
```

#### Bước 2 — Tiền xử lý dữ liệu

```bash
# Tập Train
.venv/bin/python cc_utils/preprocess_shtech.py \
    --data_dir ~/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech \
    --output_dir ./DuLieuDaXuLy \
    --dataset part_A --mode train \
    --image_size 256 --ndevices 1 --sigma '0.5' --kernel_size '3'

# Tập Test
.venv/bin/python cc_utils/preprocess_shtech.py \
    --data_dir ~/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech \
    --output_dir ./DuLieuDaXuLy \
    --dataset part_A --mode test \
    --image_size 256 --ndevices 1 --sigma '0.5' --kernel_size '3'
```

#### Bước 3 — Tải Pretrained Weights & Huấn luyện

```bash
# Tải pretrained upsampler (OpenAI)
mkdir -p weights
python3 -c "
import urllib.request
urllib.request.urlretrieve(
    'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt',
    'weights/64_256_upsampler.pt'
)
print('Tải xong!')
"

# Huấn luyện mô hình
PYTHONPATH=. .venv/bin/python scripts/super_res_train.py \
    --data_dir ./DuLieuDaXuLy/part_A/part_1/train \
    --val_samples_dir ./DuLieuDaXuLy/part_A/part_1/test \
    --log_dir ./results \
    --normalizer 0.8 --pred_channels 1 --batch_size 1 \
    --save_interval 5 --lr 1e-4 \
    --attention_resolutions 16 --class_cond False \
    --diffusion_steps 1000 --large_size 256 --small_size 256 \
    --learn_sigma True --noise_schedule linear \
    --num_channels 32 --num_head_channels 16 --num_res_blocks 1 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm True
```

#### Bước 4 — Suy diễn (Inference)

```bash
# Thay model000020.pt bằng checkpoint mới nhất của bạn
PYTHONPATH=. .venv/bin/python scripts/super_res_sample.py \
    --data_dir ./DuLieuDaXuLy/part_A/part_1/test \
    --log_dir ./results \
    --model_path ./results/model000020.pt \
    --normalizer 0.8 --pred_channels 1 --batch_size 1 --per_samples 1 \
    --attention_resolutions 16 --class_cond False \
    --diffusion_steps 100 --timestep_respacing "100" \
    --large_size 256 --small_size 256 \
    --learn_sigma True --noise_schedule linear \
    --num_channels 32 --num_head_channels 16 --num_res_blocks 1 \
    --resblock_updown True --use_fp16 False --use_scale_shift_norm True
```

#### Bước 5 — Đánh giá & Xuất Heatmap

```bash
.venv/bin/python cc_utils/evaluate.py \
    --data_dir ./DuLieuDaXuLy/part_A/part_1/test \
    --result_dir ./results \
    --output_dir ./results/evaluation \
    --image_size 256
```

**Kết quả xuất ra:**
- `results/evaluation/` — ảnh so sánh (grayscale)
- `results/evaluation/heatmaps/` — heatmap overlay màu JET (3 panel)
- `results/evaluation/scatter_pred_vs_gt.png` — scatter plot MAE

#### Bước 6 — Trực quan hóa Tùy biến (Custom Visualization)

Bạn có thể dùng công cụ độc lập của đồ án để lưu xuất riêng các hình ảnh Heatmap hoặc tạo các điểm (Green dots) trực tiếp vào ảnh gốc phục vụ cho báo cáo/luận văn:

```bash
# Vẽ Ảnh Gốc + Heatmap Overlay VÀ Ảnh Gốc + Điểm Dự Đoán Đỏ (Peaks)
.venv/bin/python scripts/visualize_results.py \
    --image DuLieuDaXuLy/part_A/part_1/test/1-1.jpg \
    --den DuLieuDaXuLy/part_A/part_1/test_den/1-1.csv \
    --out_dir results_vis

# Vẽ Ảnh Gốc + Điểm xanh Ground Truth chuẩn xác (từ file .mat)
.venv/bin/python scripts/visualize_results.py \
    --image ~/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech/part_A/test_data/images/IMG_1.jpg \
    --mat ~/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat \
    --out_dir results_vis
```

---

### 6.3 Chạy trên Cloud (Colab & Kaggle)

Vì mô hình Diffusion đòi hỏi tính toán lớn, bạn nên đưa dự án này lên các nền tảng đám mây để tận dụng GPU mạnh mẽ và miễn phí. Dự án đã cung cấp sẵn 2 file Notebook rất tiện lợi: `CrowdDiff_Colab.ipynb` và `CrowdDiff_Kaggle.ipynb` ngay trong thư mục gốc.

**Sử dụng với Google Colab:**
1. Nén toàn bộ mã nguồn của bạn thành file `crowddiff_colab.zip` (Loại trừ các thư mục `.venv`, `DuLieuDaXuLy`, `results`, `.git` để nén cực nhẹ).
2. Tải file zip lên Google Drive của bạn.
3. Mở file `CrowdDiff_Colab.ipynb` bằng Colab. Bật loại thời gian chạy là GPU (T4), thiết lập kết nối tới Google Drive và lần lượt thực thi các khối lệnh đã được hướng dẫn để giải nén Code, cài đặt thư viện và tự động Train.

**Sử dụng với Kaggle Notebook (⭐ Rất khuyên dùng):**
1. Đăng nhập Kaggle và tạo một Dataset mới (ví dụ đặt tên là `crowddiff-source`). Tải file nén `crowddiff_colab.zip` lên Dataset đó (Kaggle sẽ tự động giải nén).
2. Tạo một Notebook mới trên Kaggle, chọn menu `File -> Import Notebook` và tải lên file `CrowdDiff_Kaggle.ipynb`.
3. Bấm **Add Data** ở cột bên phải, tìm và gắn thêm Dataset `crowddiff-source` (vừa tạo) và Dataset `tthien/shanghaitech` vào không gian làm việc.
4. Ở thẻ *Session Options*, bật **Internet** chuyển sang On, chọn Accelerator là **GPU P100** hoặc **GPU T4x2**.
5. Bấm Play chạy tuần tự các lệnh để sao chép code. Đặc biệt, để train qua đêm mà không bị mất mạng, ở Bước 5 hãy bấm **Save Version -> Save & Run All (Commit)** và tắt máy tính đi ngủ, Kaggle sẽ tự động chạy ngầm trên máy chủ và nén kết quả để bạn tải về vào buổi sáng!

---

## 7. Phân tích kết quả — Notebooks

```bash
# Mở Jupyter Notebook
.venv/bin/python -m jupyter notebook notebooks/
```

| Notebook | Nội dung |
|---|---|
| `01_explore_dataset.ipynb` | Thống kê dataset · Histogram số người · Heatmap GT · So sánh thưa vs dày |
| `03_evaluate_results.ipynb` | Bảng MAE/MSE · Scatter plot · Heatmap overlay kết quả · Top tốt/tệ nhất |

---

## 8. Kết quả & Đánh giá

### Ví dụ kết quả (multi-hypothesis)

<p float="left">
  <img src="figs/trial3 356.jpg" width="200"/>
  <img src="figs/trial4 360.jpg" width="200"/>
  <img src="figs/gt 361.jpg" width="200"/>
  <img src="figs/final 359.jpg" width="200"/>
</p>

*Trial 3: 356 · Trial 4: 360 · Ground Truth: 361 · Kết quả cuối: 359*

### Thang đo sai số

| Chỉ số | Ý nghĩa |
|---|---|
| **MAE** (Mean Absolute Error) | Sai số tuyệt đối trung bình giữa số người dự đoán và thực tế |
| **MSE** (Mean Square Error / RMSE) | Đo độ phân tán của sai số, nhạy cảm hơn với các trường hợp sai nhiều |

### So sánh tham khảo (ShanghaiTech Part A)

| Phương pháp | MAE | MSE |
|---|---|---|
| MCNN (2016) | 110.2 | 173.2 |
| CSRNet (2018) | 68.2 | 115.0 |
| DM-Count (2020) | 59.7 | 95.7 |
| **CrowdDiff (paper)** | **53.7** | **77.1** |

> *Kết quả của bạn phụ thuộc vào số bước huấn luyện và cấu hình mô hình.*

---

## 9. Tài liệu tham khảo

1. **CrowdDiff** — *Diffuse-Denoise-Count: Accurate Crowd Counting with Diffusion Models* ([Paper](https://arxiv.org/abs/2303.12790))
2. **OpenAI Guided Diffusion** — Codebase gốc ([GitHub](https://github.com/openai/guided-diffusion))
3. **ShanghaiTech Dataset** — *Single-Image Crowd Counting via Multi-Column CNN* (CVPR 2016)
4. **DDPM** — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
5. **CSRNet** — *Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes* (CVPR 2018)

---

*Phát triển dựa trên codebase [guided-diffusion](https://github.com/openai/guided-diffusion) của OpenAI.*
