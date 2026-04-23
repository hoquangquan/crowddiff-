<div align="center">

# CrowdDiff: Ước tính Mật độ Đám đông Đa Giả thuyết sử dụng Mô hình Khuếch tán (Diffusion Models)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.7-blue.svg" alt="Python 3.9" />
  <img src="https://img.shields.io/badge/pytorch-1.13.1-red.svg" alt="PyTorch 1.13.1" />
  <a href="https://github.com/openai/guided-diffusion"><img src="https://img.shields.io/badge/code%20base-guided--diffusion-green.svg" alt="Guided Diffusion" /></a>
</p>

Kho lưu trữ này chứa mã nguồn để triển khai bằng PyTorch cho bài báo [**Diffuse-Denoise-Count: Accurate Crowd Counting with Diffusion Models**](#-trích-dẫn) (CVPR 2024).

</div>

---

## 📖 Phương pháp
<img src="figs/flow chart.jpg" width="1000" alt="Flow Chart"/> 

## 👁️ Các bản trình diễn trực quan

### Bản đồ mật độ (Density Map)
<p float="left">
  <img src="figs/jhu 01.gif" width="400" height="245"/>
  <img src="figs/jhu 02.gif" width="400" height="245"/>
  <br>
  <img src="figs/shha.gif" width="400" height="245"/>
  <img src="figs/ucf qnrf.gif" width="400" height="245"/>
</p>

### Bản đồ đám đông và Sinh ngẫu nhiên (Stochastic Generation)
<p float="left">
  <img src="figs/gt 361.jpg" width="263" height="172"/>
  <img src="figs/trial1 349.jpg" width="263" height="172"/>
  <img src="figs/trial2 351.jpg" width="263" height="172"/>
</p>
<p align="left">
  <em>&emsp; &emsp; &emsp; Thực tế (Ground Truth): 361 &emsp; &emsp; Lần thử 1: 349 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Lần thử 2: 351</em>
</p>

<p float="left">
  <img src="figs/final 359.jpg" width="263" height="172"/>
  <img src="figs/trial3 356.jpg" width="263" height="172"/>
  <img src="figs/trial4 360.jpg" width="263" height="172"/>
</p>
<p align="left">
  <em>&emsp; &emsp; &emsp; Dự đoán cuối cùng: 359 &emsp; &emsp; &emsp; &emsp; Lần thử 3: 356 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Lần thử 4: 360</em>
</p>

## ⚙️ Cài đặt

Cài đặt các thư viện Python theo yêu cầu. Đoạn mã này được kiểm tra chủ yếu bằng Python 3.9.7 và PyTorch 1.13.1.
```bash
pip install -r requirements.txt
```

## 📂 Chuẩn bị dữ liệu

Chạy tập lệnh tiền xử lý để chuẩn bị tập dữ liệu:
```bash
python cc_utils/preprocess_shtech.py \
    --data_dir path/to/data \
    --output_dir path/to/save \
    --dataset dataset \
    --mode test \
    --image_size 256 \
    --ndevices 1 \
    --sigma '0.5' \
    --kernel_size '3'
```

## 🚀 Huấn luyện (Training)

1. Tải về [trọng số đã được huấn luyện sẵn (pre-trained weights)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt).
2. Chạy tập lệnh huấn luyện:

```bash
DATA_DIR="--data_dir path/to/train/data --val_samples_dir path/to/val/data"
LOG_DIR="--log_dir path/to/results --resume_checkpoint path/to/pre-trained/weights"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 8 --save_interval 10000 --lr 1e-4"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python scripts/super_res_train.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

## 🧪 Kiểm thử (Testing)

1. Tải về [trọng số mô hình để kiểm thử (testing model weights)](https://drive.google.com/file/d/1dLEjaZqw9bxQm2sUU4I6YXDnFfyEHl8p/view?usp=sharing).
2. Chạy tập lệnh kiểm thử:
```bash
DATA_DIR="--data_dir path/to/test/data"
LOG_DIR="--log_dir path/to/results --model_path path/to/model"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 1 --per_samples 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python scripts/super_res_sample.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

## 📜 Trích dẫn

Nếu bạn thấy dự án này có ích cho công việc hoặc nghiên cứu của mình, vui lòng sử dụng trích dẫn (BibTeX) sau:
```bibtex
@inproceedings{ranasinghe2024diffuse,
  title={Diffuse-Denoise-Count: Accurate Crowd-Counting with Diffusion Models},
  author={Ranasinghe, Yasiru and Nair, Nithin Gopalakrishnan and Bandara, Wele Gedara Chaminda and Patel, Vishal M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23293--23302},
  year={2024}
}
```

## 🙏 Lời cảm ơn

Một phần mã nguồn ứng dụng trong dự án được tham khảo và mở rộng từ kho lưu trữ [guided-diffusion](https://github.com/openai/guided-diffusion) của OpenAI.
