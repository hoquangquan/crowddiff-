#!/bin/bash
# ==============================================================================
# run_pipeline.sh — Pipeline Đầy Đủ: CrowdDiff cho ShanghaiTech Part A
# ==============================================================================
# Chạy toàn bộ quy trình từ tải dữ liệu → tiền xử lý → huấn luyện → suy diễn → đánh giá
#
# Cách dùng:
#   bash run_pipeline.sh [--steps STEP1,STEP2,...] [--model_path PATH]
#
# Các bước có thể chọn:
#   download   : Tải dataset từ Kaggle Hub
#   preprocess : Tiền xử lý ảnh + density map
#   train      : Huấn luyện mô hình diffusion
#   sample     : Suy diễn / sinh density map
#   evaluate   : Đánh giá MAE, MSE + xuất heatmap
#   all        : Chạy tất cả các bước (mặc định)
#
# Ví dụ:
#   bash run_pipeline.sh                             # Chạy toàn bộ
#   bash run_pipeline.sh --steps download,preprocess # Chỉ tải + tiền xử lý
#   bash run_pipeline.sh --steps sample,evaluate --model_path results/model000020.pt
# ==============================================================================

set -e  # Dừng ngay nếu có lỗi

# ────────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ────────────────────────────────────────────────────────────────────────────────
PYTHON=".venv/bin/python"
PYTHONPATH_PREFIX="PYTHONPATH=."

# Đường dẫn
KAGGLE_DATA_DIR="$HOME/.cache/kagglehub/datasets/tthien/shanghaitech/versions/1/ShanghaiTech"
PROCESSED_DIR="./DuLieuDaXuLy"
RESULTS_DIR="./results"
WEIGHTS_DIR="./weights"
LOG_DIR="./results"

# Tham số huấn luyện (light mode — phù hợp GPU nhỏ)
IMAGE_SIZE=256
BATCH_SIZE=1
SAVE_INTERVAL=1000
LR=1e-4
DIFFUSION_STEPS=1000
NUM_CHANNELS=32
NUM_RES_BLOCKS=1

# Tham số suy diễn
MODEL_PATH=""           # Để trống để tự tìm model mới nhất
INFERENCE_STEPS=100
PER_SAMPLES=1
NORMALIZER=0.8

# Bước mặc định
STEPS="all"

# ────────────────────────────────────────────────────────────────────────────────
# PARSE THAM SỐ
# ────────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --diffusion_steps)
            DIFFUSION_STEPS="$2"
            shift 2
            ;;
        --num_channels)
            NUM_CHANNELS="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "❌ Tham số không hợp lệ: $1"
            exit 1
            ;;
    esac
done

# ────────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ────────────────────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  🔷 $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_step() {
    local STEP="$1"
    if [[ "$STEPS" == "all" || "$STEPS" == *"$STEP"* ]]; then
        return 0  # Chạy bước này
    fi
    return 1     # Bỏ qua
}

find_latest_model() {
    # Tìm model checkpoint mới nhất trong thư mục results
    latest=$(ls -t "$RESULTS_DIR"/model*.pt 2>/dev/null | head -1)
    if [[ -z "$latest" ]]; then
        echo "❌ Không tìm thấy model checkpoint trong $RESULTS_DIR"
        exit 1
    fi
    echo "$latest"
}

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 0: KHỞI ĐẦU
# ────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   CrowdDiff — Pipeline Đếm Đám Đông · ShanghaiTech Part A          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📌 Thư mục gốc  : $(pwd)"
echo "📌 Python       : $PYTHON"
echo "📌 Bước chạy    : $STEPS"
echo "📌 Thời gian    : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

mkdir -p "$PROCESSED_DIR" "$RESULTS_DIR" "$WEIGHTS_DIR"

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: TẢI DATASET
# ────────────────────────────────────────────────────────────────────────────────
if run_step "download"; then
    print_header "BƯỚC 1/5: Tải Dataset ShanghaiTech Part A từ Kaggle Hub"
    
    $PYTHON -c "
import kagglehub
print('⏳ Đang tải dataset...')
path = kagglehub.dataset_download('tthien/shanghaitech')
print(f'✅ Dataset đã tải về: {path}')
"
fi

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: TIỀN XỬ LÝ
# ────────────────────────────────────────────────────────────────────────────────
if run_step "preprocess"; then
    print_header "BƯỚC 2/5: Tiền xử lý dữ liệu (tạo crops + density CSV)"
    
    echo "📊 Xử lý tập TRAIN..."
    $PYTHON cc_utils/preprocess_shtech.py \
        --data_dir "$KAGGLE_DATA_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --dataset part_A \
        --mode train \
        --image_size $IMAGE_SIZE --ndevices 1 --sigma '0.5' --kernel_size '3'

    echo ""
    echo "📊 Xử lý tập TEST..."
    $PYTHON cc_utils/preprocess_shtech.py \
        --data_dir "$KAGGLE_DATA_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --dataset part_A \
        --mode test \
        --image_size $IMAGE_SIZE --ndevices 1 --sigma '0.5' --kernel_size '3'

    echo "✅ Tiền xử lý hoàn tất → $PROCESSED_DIR/part_A/"
fi

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: HUẤN LUYỆN
# ────────────────────────────────────────────────────────────────────────────────
if run_step "train"; then
    print_header "BƯỚC 3/5: Huấn luyện Mô hình Diffusion"
    
    # Tải trọng số khởi điểm nếu chưa có
    if [[ ! -f "$WEIGHTS_DIR/64_256_upsampler.pt" ]]; then
        echo "⏳ Đang tải pretrained upsampler weights..."
        $PYTHON -c "
import urllib.request, os
os.makedirs('$WEIGHTS_DIR', exist_ok=True)
url = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt'
urllib.request.urlretrieve(url, '$WEIGHTS_DIR/64_256_upsampler.pt')
print('✅ Đã tải trọng số: $WEIGHTS_DIR/64_256_upsampler.pt')
"
    fi

    TRAIN_DIR="$PROCESSED_DIR/part_A/part_1/train"
    VAL_DIR="$PROCESSED_DIR/part_A/part_1/test"

    eval "PYTHONPATH=. $PYTHON scripts/super_res_train.py \
        --data_dir $TRAIN_DIR \
        --val_samples_dir $VAL_DIR \
        --log_dir $LOG_DIR \
        --normalizer $NORMALIZER \
        --pred_channels 1 \
        --batch_size $BATCH_SIZE \
        --save_interval $SAVE_INTERVAL \
        --lr $LR \
        --attention_resolutions 16 \
        --class_cond False \
        --diffusion_steps $DIFFUSION_STEPS \
        --large_size $IMAGE_SIZE \
        --small_size $IMAGE_SIZE \
        --learn_sigma True \
        --noise_schedule linear \
        --num_channels $NUM_CHANNELS \
        --num_head_channels 16 \
        --num_res_blocks $NUM_RES_BLOCKS \
        --resblock_updown True \
        --use_fp16 True \
        --use_scale_shift_norm True"

    echo "✅ Huấn luyện hoàn tất → $LOG_DIR/"
fi

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: SUY DIỄN (INFERENCE)
# ────────────────────────────────────────────────────────────────────────────────
if run_step "sample"; then
    print_header "BƯỚC 4/5: Suy diễn — Sinh Density Map từ Model"

    # Tự động tìm model mới nhất nếu không chỉ định
    if [[ -z "$MODEL_PATH" ]]; then
        MODEL_PATH=$(find_latest_model)
        echo "🔍 Sử dụng model: $MODEL_PATH"
    fi

    TEST_DIR="$PROCESSED_DIR/part_A/part_1/test"

    eval "PYTHONPATH=. $PYTHON scripts/super_res_sample.py \
        --data_dir $TEST_DIR \
        --log_dir $LOG_DIR \
        --model_path $MODEL_PATH \
        --normalizer $NORMALIZER \
        --pred_channels 1 \
        --batch_size 1 \
        --per_samples $PER_SAMPLES \
        --attention_resolutions 16 \
        --class_cond False \
        --diffusion_steps $INFERENCE_STEPS \
        --timestep_respacing \"$INFERENCE_STEPS\" \
        --large_size $IMAGE_SIZE \
        --small_size $IMAGE_SIZE \
        --learn_sigma True \
        --noise_schedule linear \
        --num_channels $NUM_CHANNELS \
        --num_head_channels 16 \
        --num_res_blocks $NUM_RES_BLOCKS \
        --resblock_updown True \
        --use_fp16 False \
        --use_scale_shift_norm True"

    echo "✅ Suy diễn hoàn tất → $LOG_DIR/"
fi

# ────────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: ĐÁNH GIÁ
# ────────────────────────────────────────────────────────────────────────────────
if run_step "evaluate"; then
    print_header "BƯỚC 5/5: Đánh giá — Tính MAE/MSE + Xuất Heatmap"

    EVAL_OUTPUT="./results/evaluation"
    mkdir -p "$EVAL_OUTPUT"

    $PYTHON cc_utils/evaluate.py \
        --data_dir "$PROCESSED_DIR/part_A/part_1/test" \
        --result_dir "$LOG_DIR" \
        --output_dir "$EVAL_OUTPUT" \
        --image_size $IMAGE_SIZE

    echo ""
    echo "✅ Đánh giá hoàn tất → $EVAL_OUTPUT/"
    echo ""
    echo "💡 Mở notebook để xem kết quả đầy đủ:"
    echo "   jupyter notebook notebooks/03_evaluate_results.ipynb"
fi

# ────────────────────────────────────────────────────────────────────────────────
# HOÀN THÀNH
# ────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   ✅ PIPELINE HOÀN TẤT                                              ║"
echo "║   📁 Kết quả: $LOG_DIR                                              ║"
echo "║   📁 Heatmap: $LOG_DIR/evaluation                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
