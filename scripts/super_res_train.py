"""
Train a super-resolution model.
Huấn luyện một mô hình Siêu độ phân giải (Super-resolution). 
Kịch bản này là script chính để train model CrowdDiff.
"""

import argparse
import glob
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch as th

from time import time, sleep

import torch.nn.functional as F
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    # Khởi tạo argparser và parse các tham số đầu vào
    args = create_argparser().parse_args()

    # Thiết lập môi trường phân tán (distributed) để có thể chạy trên nhiều GPU/node
    dist_util.setup_dist()
    # Cấu hình thư mục lưu log và định dạng output
    logger.configure(dir=args.log_dir)#, format_strs=['stdout', 'wandb'])

    logger.log("creating model...")

    # Khởi tạo mạng U-Net (model) và thuật toán khuếch tán Diffusion.
    # Hàm này tự động chọn cấu hình phù hợp bằng cách kết hợp mặc định và đối số được truyền vào.
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    # Đưa model lên thiết bị tính toán (GPU hoặc CPU) do môi trường phân tán quyết định
    model.to(dist_util.dev())
    # Lịch trình lấy mẫu timestep (chọn thời điểm trong quá trình Diffusion)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Chuyển tham số dạng chuỗi '0.2,0.3' thành danh sách [0.2, 0.3] để chuẩn hoá mật độ sau này
    args.normalizer = [float(value) for value in args.normalizer.split(',')]
    # Chú thích: Code sau đây (đã comment) dường như dùng để gán nhãn thể loại (condition) dạng chuỗi sang id
    # args.num_classes = [str(index) for index in range(args.num_classes)]
    # args.num_classes = sorted(args.num_classes)
    # args.num_classes = {k: i for i,k in enumerate(args.num_classes)}

    logger.log("creating data loader...")
    # Tải dữ liệu dùng để huấn luyện:
    # Bao gồm các cặp ảnh có độ phân giải thấp (ảnh đám đông) và các bản đồ mất độ cần khôi phục (Siêu độ phân giải)
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        normalizer=args.normalizer,
        pred_channels=args.pred_channels,
    )
    # Tương tự cho validation, đã comment bản không sử dụng "args" trực tiếp
    # val_data = load_data_for_worker(args.val_samples_dir,args.val_batch_size, args.normalizer, args.pred_channels,
    #                                 args.num_classes, class_cond=True)
    val_data = load_data_for_worker(args)

    logger.log("training...")
    # Khởi tạo và chạy Vòng lặp huấn luyện chính. Quản lý tối ưu hoá (Step), theo dõi ema (Exponential moving average) v.v..
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        normalizer=args.normalizer,
        pred_channels=args.pred_channels,
        base_samples=args.val_samples_dir,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,           # Tốc độ duy trì đường trung bình hàm mũ (giúp trọng số mượt hơn)
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,           # Bật Mixed Precision FP16
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps, # Tham số giảm learning rate sau mỗi số bước
    ).run_loop()


def load_superres_data(data_dir, batch_size, large_size, small_size, normalizer, pred_channels, class_cond=False):
    """
    Hàm yield/tạo chuỗi generator sinh dữ liệu huấn luyện Super Resolution.
    Trả về bộ ảnh đích (ground_truth high resolution) và dictionary chứa tham số mô hình, 
    trong đó 'low_res' làm điều kiện khuếch tán.
    """
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        normalizer=normalizer,
        pred_channels=pred_channels,
    )
    for large_batch, model_kwargs in data:
        # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        # Cách đang dùng: Dữ liệu (large_batch) đọc ra được gộp chung ảnh mật độ và ảnh thực tế.
        # large_batch mới (số kênh từ 0 -> pred_channels) sẽ là bản đồ mật độ (density/high_res).
        # model_kwargs["low_res"] (các kênh còn lại) là ảnh nguồn cần tăng chất lượng hoặc dùng làm dẫn đường (guidance proxy).
        large_batch, model_kwargs["low_res"] = large_batch[:,:pred_channels], large_batch[:,pred_channels:]
        yield large_batch, model_kwargs


# def load_data_for_worker(base_samples, batch_size, normalizer, pred_channels, class_cond=False):
def load_data_for_worker(args):
    """
    Đọc dữ liệu kiểm tra validation cho hệ thống đa luồng (worker).
    """
    base_samples, batch_size, normalizer, pred_channels = args.val_samples_dir, args.val_batch_size, args.normalizer, args.pred_channels
    class_labels, class_cond = args.num_classes, args.class_cond
    # start = time()
    # Tìm kiếm các ảnh JPG ở thư mục validation
    img_list = glob.glob(os.path.join(base_samples,'*.jpg'))
    img_list = img_list
    den_list = []
    # Khớp để tìm các file csv (density map dạng ma trận csv) tương ứng với mỗi ảnh
    for _ in img_list:
        den_path =  _.replace('test','test_den')
        den_path = den_path.replace('.jpg','.csv')
        den_list.append(den_path)
    # print(f'list prepared: {(time()-start) :.4f}s.')

    image_arr, den_arr = [], []
    for file in img_list:
        # start = time()
        # Đọc ảnh gốc vào mảng
        image = Image.open(file)
        image_arr.append(np.asarray(image))
        # print(f'image read: {(time()-start) :.4f}s.')

        # start = time()
        # Đọc tham số mật độ từ tập tin
        file = file.replace('test','test_den').replace('jpg','csv')
        image = np.asarray(pd.read_csv(file, header=None).values)
        # print(f'density read: {(time()-start) :.4f}s.')

        # start = time()
        # Tách cấu trúc file theo normalizer rồi chuẩn hoá kích thước mật độ theo từng phân lớp normalizer
        image = np.stack(np.split(image, len(normalizer), -1))
        # Tỉ lệ hoá tương ứng (chia cho mốc max của từng dải normalizer)
        image = np.asarray([m/n for m,n in zip(image, normalizer)])
        # Cắt để nằm trong dải [0, 1] và đổi chiều thành Shape (H, W, C)
        image = image.transpose(1,2,0).clip(0,1)
        den_arr.append(image)
        # print(f'density prepared: {(time()-start) :.4f}s.')


    # Lấy thông tin về index của GPU hiện tại (rank) và số lượng tổng thiết bị (num_ranks)
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer, den_buffer = [], []
    label_buffer = []
    name_buffer = []
    
    # Một vòng lặp Generator vô tận
    while True:
        # Lặp mảng dữ liệu có nhảy bước num_ranks để luồng hiện tại chỉ lấy phân nửa/vài dữ liệu khác vs luồng khác
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i]), den_buffer.append(den_arr[i])
            name_buffer.append(os.path.basename(img_list[i]))
            
            # Nếu mô hình có tuỳ biến class class_label
            if class_cond:
                class_label = os.path.basename(img_list[i]).split('_')[0]
                class_label = class_labels[class_label]
                label_buffer.append(class_label)
                # pass
            
            # Gửi dữ liệu batch khi danh sách buffer đã đầy bằng quy mô batch (batch_size)
            if len(buffer) == batch_size:
                # Đưa mảng Numpy vào Torch Tensor FP32
                batch = th.from_numpy(np.stack(buffer)).float()
                # Ảnh gốc chuẩn hóa về biên độ [-1.0, 1.0] (ảnh gốc đang có pixel 0-255)
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                
                # Biến density map cũng thành Tensor tương tự
                den_batch = th.from_numpy(np.stack(den_buffer)).float()
                # den_batch = den_batch / normalizer 
                den_batch = 2*den_batch - 1      # Chuyển hoá mật độ [0,1] -> [-1,1]
                den_batch = den_batch.permute(0, 3, 1, 2)
                
                # Tập hợp result dict
                res = dict(low_res=batch,
                           name=name_buffer,
                           high_res=den_batch
                           )
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                
                # Trống các buffer để dùng cho lượt mới
                buffer, label_buffer, name_buffer, den_buffer = [], [], [], []


def create_argparser():
    """
    Tạo config parse tham số lệnh.
    """
    defaults = dict(
        data_dir="",
        val_batch_size=1,           # Kích thước batch khi kiểm định (validation)
        val_samples_dir=None,
        log_dir=None,               # Thư mục xuất ra báo cáo và Checkpoint mô hình
        schedule_sampler="uniform", # Thuật toán để lấy mẫu các timestep ('uniform' là bốc đều từ 0 tới timesteps)
        lr=1e-4,                    # Tham số học máy (learning rate)
        weight_decay=0.0,
        lr_anneal_steps=0,          # Giảm kích cỡ learning rate khi đạt số steps nhất định
        batch_size=1,
        microbatch=-1,              # Cắt batch ra làm microbatch trong 1 step để giảm hao VRAM (-1 là bị tắt)
        ema_rate="0.9999",
        log_interval=10,            # Sau mỗi n step thì ghi tensorboard / log in
        save_interval=10000,        # Sau một số step sẽ export weights dưới dạng file .pt ở log_dir
        resume_checkpoint="",       # Truyền param tới model có sẵn để train tiếp
        use_fp16=False,             # Bật tối ưu bộ nhớ Mixed Precision FP16
        fp16_scale_growth=1e-3,     # Cấu hình scale ban đầu (Gradient scaler) dùng trong giảm lạm/quá dòng của fp16
        normalizer='0.2',
        pred_channels=3,            # Số tham số (thường là 3 cho bản đồ crowd)
        num_classes=13,
    )
    # Tải các mục mặc định từ framework (ví dụ như channels=64, num_res_blocks=2, image_size..)
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
