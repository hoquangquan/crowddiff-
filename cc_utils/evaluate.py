import os
from PIL import Image
import argparse
import numpy as np
"""
Công cụ tính độ sai lệch MAE và MSE sử dụng mô hình dự đoán được lưu trữ ở file ảnh hệ thống.
Kết quả bao gồm: điểm số MAE/MSE toàn cục + hình ảnh heatmap overlay màu JET.
"""
import torch as th
from einops import rearrange
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter


def get_arg_parser():
    parser = argparse.ArgumentParser('Parameters for the evaluation', add_help=False)

    parser.add_argument('--data_dir', default='primary_datasets/shtech_A/test_data/images', type=str,
                        help='Path to the original image directory')
    parser.add_argument('--result_dir', default='experiments/shtech_A', type=str,
                        help='Path to the diffusion results directory')
    parser.add_argument('--output_dir', default='experiments/evaluate', type=str,
                        help='Path to the output directory')
    parser.add_argument('--image_size', default=256, type=int,
                        help='Crop size')

    return parser


def config(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def main(args):
    """
    Quy trình đánh giá (Evaluation) các mẫu:
    Thiết lập path, khâu/nối các ảnh kết quả crop, bóc tách điểm density và báo cáo giá trị trung bình MAE, MSE toàn cục.
    """
    data_dir = args.data_dir
    result_dir = args.result_dir
    output_dir = args.output_dir
    image_size = args.image_size

    config(output_dir)
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    config(heatmap_dir)

    img_list = sorted(os.listdir(data_dir))
    result_list = os.listdir(result_dir)

    mae, mse = 0, 0
    all_preds, all_gts = [], []
    per_image_errors = []

    for index, name in enumerate(img_list):
        image = Image.open(os.path.join(data_dir, name)).convert('RGB')

        try:
            crops, gt_count = get_crops(result_dir, name.split('_')[-1], image, result_list)
        except Exception:
            continue

        pred = crops[:,:, image_size:-image_size,:].mean(-1)
        gt = crops[:,:, -image_size:,:].mean(-1)
        
        pred = remove_background(pred)

        pred = combine_crops(pred, image, image_size)
        gt = combine_crops(gt, image, image_size)

        pred_count = get_circle_count(pred)

        # ── Xuất heatmap màu JET overlay ───────────────────────
        image_np = np.asarray(image)
        heatmap_path = os.path.join(heatmap_dir, name.replace('.jpg', '.png'))
        save_heatmap_overlay(image_np, pred, gt, pred_count, gt_count, heatmap_path)

        # ── Lưu ảnh so sánh đơn giản (grayscale) ──────────────
        pred_3ch = np.repeat(pred[:,:,np.newaxis],3,-1)
        gt_3ch   = np.repeat(gt[:,:,np.newaxis],3,-1)
        image_np_copy = image_np.copy()

        gap = 5
        red_gap = np.zeros((image_np_copy.shape[0],gap,3), dtype=np.uint8)
        red_gap[:,:,0] = 255

        compare = np.concatenate([image_np_copy, red_gap, pred_3ch, red_gap, gt_3ch], axis=1)
        cv2.imwrite(os.path.join(output_dir, name), compare[:,:,::-1])

        err = abs(pred_count - gt_count)
        mae += err
        mse += err ** 2
        all_preds.append(pred_count)
        all_gts.append(gt_count)
        per_image_errors.append({'name': name, 'pred': pred_count, 'gt': int(gt_count), 'abs_err': err})
        print(f'[{index+1:3d}] {name:30s}  pred={pred_count:5.0f}  gt={gt_count:5.0f}  err={err:.0f}')

    n = len(per_image_errors)
    final_mae = mae / n if n > 0 else 0
    final_mse = np.sqrt(mse / n) if n > 0 else 0
    print()
    print('═' * 55)
    print(f'  Tổng ảnh đánh giá : {n}')
    print(f'  MAE (Mean Abs Err): {final_mae:.2f}')
    print(f'  MSE (Root Mean Sq): {final_mse:.2f}')
    print('═' * 55)
    print(f'📁 Heatmap lưu tại : {heatmap_dir}')
    print()

    # Vẽ scatter plot Predicted vs Ground Truth
    _save_scatter_plot(all_preds, all_gts, output_dir, final_mae, final_mse)


def save_heatmap_overlay(image_np, pred_density, gt_density, pred_count, gt_count, save_path):
    """
    Tạo và lưu hình ảnh 3 panel:
      [Ảnh gốc] | [Heatmap dự đoán (JET overlay)] | [Heatmap Ground Truth (JET overlay)]
    Mỗi panel có tiêu đề với số lượng người.
    """
    def make_heatmap_overlay(base_img, density_map, alpha=0.55):
        """Overlay density map lên ảnh gốc với colormap JET."""
        # Làm mượt density map bằng Gaussian filter
        smooth = gaussian_filter(density_map.astype(np.float32), sigma=4)
        # Chuẩn hoá về [0, 1]
        dmax = smooth.max()
        if dmax > 0:
            smooth = smooth / dmax
        # Áp colormap JET
        colormap = plt.cm.jet(smooth)[:, :, :3]  # (H, W, 3) float64
        colormap = (colormap * 255).astype(np.uint8)
        # Alpha blend
        overlay = (alpha * colormap + (1 - alpha) * base_img).astype(np.uint8)
        return overlay

    h, w = image_np.shape[:2]
    gap = np.ones((h, 6, 3), dtype=np.uint8) * 200  # separator màu xám nhạt

    pred_overlay = make_heatmap_overlay(image_np, pred_density)
    gt_overlay   = make_heatmap_overlay(image_np, gt_density)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')

    titles = [
        ('Ảnh Gốc', image_np, None),
        (f'Dự đoán: {int(pred_count)} người', pred_overlay, 'jet'),
        (f'Ground Truth: {int(gt_count)} người', gt_overlay, 'jet'),
    ]

    for ax, (title, img, cmap) in zip(axes, titles):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')
        # Viền màu cho panel
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    error = abs(pred_count - gt_count)
    fig.suptitle(
        f'Sai số (MAE): {error:.0f} người',
        color='#f0c040', fontsize=14, fontweight='bold', y=0.02
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_scatter_plot(preds, gts, output_dir, mae, mse):
    """Vẽ scatter plot: Predicted Count vs Ground Truth Count."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    gts_arr   = np.array(gts)
    preds_arr = np.array(preds)

    ax.scatter(gts_arr, preds_arr, alpha=0.7, color='#5bc8f5', edgecolors='white', linewidths=0.4, s=60)
    lim = max(gts_arr.max(), preds_arr.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('Ground Truth (số người)', color='white', fontsize=12)
    ax.set_ylabel('Dự đoán (số người)',      color='white', fontsize=12)
    ax.set_title(f'Predicted vs Ground Truth\nMAE={mae:.2f}  MSE={mse:.2f}',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(facecolor='#2a2a4e', labelcolor='white')

    scatter_path = os.path.join(output_dir, 'scatter_pred_vs_gt.png')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'📊 Scatter plot lưu tại: {scatter_path}')


def remove_background(crops):
    def count_colors(image):

        colors_count = {}
        # Flattens the 2D single channel array so as to make it easier to iterate over it
        image = image.flatten()
        # channel_g = channel_g.flatten()  # ""
        # channel_r = channel_r.flatten()  # ""

        for i in range(len(image)):
            I = str(int(image[i]))
            if I in colors_count:
                colors_count[I] += 1
            else:
                colors_count[I] = 1
        
        return int(max(colors_count, key=colors_count.__getitem__))+5

    for index, crop in enumerate(crops):
        count = count_colors(crop)
        crops[index] = crop*(crop>count)

    return crops


def get_crops(path, index, image, result_list, image_size=256):
    w, h = image.size
    ncrops = ((h-1+image_size)//image_size)*((w-1+image_size)//image_size)
    crops = []

    gt_count = 0
    for _ in range(ncrops):
        crop = f'{index.split(".")[0]}-{_+1}'
        for _ in result_list:
            if _.startswith(crop):
                break

        crop = Image.open(os.path.join(path,_))
        # crop = Image.open()
        crops.append(np.asarray(crop))
        gt_count += float(_.split(' ')[-1].split('.')[0])
    crops = np.stack(crops)
    if len(crops.shape) < 4:
        crops = np.expand_dims(crops, 0)
    
    return crops, gt_count
    

def combine_crops(density, image, image_size):
    w,h = image.size
    p1 = (h-1+image_size)//image_size
    density = th.from_numpy(density)
    density = rearrange(density, '(p1 p2) h w-> (p1 h) (p2 w)', p1=p1)
    den_h, den_w = density.shape

    start_h, start_w = (den_h-h)//2, (den_w-w)//2
    end_h, end_w = start_h+h, start_w+w
    density = density[start_h:end_h, start_w:end_w]
    # print(density.max(), density.min())
    # density = density*(density>0)
    # assert False
    return density.numpy().astype(np.uint8)


def get_circle_count(image, threshold=0, draw=False):

    # Denoising
    denoisedImg = cv2.fastNlMeansDenoising(image)

    # Threshold (binary image)
    # thresh – threshold value.
    # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    # type – thresholding type
    th, threshedImg = cv2.threshold(denoisedImg, threshold, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

    # Perform morphological transformations using an erosion and dilation as basic operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

        Image.fromarray(contoursImg, mode='RGB').show()

    return max(len(contours)-1,0) # remove the outerboarder countour


# def get_circle_count_and_sample(samples, thresh=0):

    count = [], []
    for sample in samples:
        pred_count = get_circle_count(sample. thresh)
        mae.append(th.abs(pred_count-gt_count))
        count.append(th.tensor(pred_count))
    
    mae = th.stack(mae)
    count = th.stack(count)

    index = th.argmin(mae)

    return index, mae[index], count[index], gt_count


if __name__=='__main__':
    parser = argparse.ArgumentParser('Combine the results and evaluate', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)