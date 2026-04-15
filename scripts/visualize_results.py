import cv2
import numpy as np
import scipy.io as io
import argparse
import os
from skimage.feature import peak_local_max
import pandas as pd

def draw_gt_points(image_path, mat_path, output_path):
    """
    Tải một bức ảnh và file điểm Ground Truth (định dạng MATLAB) của nó,
    tiến hành vẽ các tọa độ điểm này lên ảnh và lưu kết quả.
    """
    # Đọc ảnh gốc bằng OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")
    
    # Đọc nội dung file .mat chức tọa độ Ground Truth
    mat = io.loadmat(mat_path)
    
    # Trích xuất tọa độ điểm. 
    # Cấu trúc dữ liệu tiêu chuẩn của tập dữ liệu ShanghaiTech Part A được lồng trong 'image_info{1,1}.location'
    try:
        points = mat['image_info'][0, 0][0, 0][0]
    except KeyError:
        print(f"Cảnh báo: Không thể phân tích cấu trúc ShanghaiTech chuẩn từ file {mat_path}. Vui lòng kiểm tra lại cấu trúc file.")
        return

    # Duyệt qua từng tọa độ và vẽ các vòng tròn màu xanh lá cây (green circles) để biểu thị
    for p in points:
        x, y = int(p[0]), int(p[1])
        # cv2.circle(ảnh, tọa độ_tâm, bán_kính, màu_sắc_BGR, độ_dày_viền) (thickness=-1 nghĩa là tô kín hình tròn)
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Lưu ảnh đã vẽ ra file
    cv2.imwrite(output_path, image)
    print(f"[GT Points] Đã lưu ảnh chồng Ground Truth tại: {output_path}")

def find_and_draw_peaks(image_path, density_map_path, output_path):
    """
    Tải ảnh gốc và bản đồ mật độ (density map) dự đoán tương ứng (có thể là file .npy hoặc .csv).
    Tìm các điểm cực đại trên bản đồ tính toán (đại diện cho vị trí từng người/blob),
    sau đó vẽ chúng lên ảnh gốc.
    """
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")

    # Đọc ma trận mật độ dự đoán từ file
    if density_map_path.endswith('.npy'):
        density_map = np.load(density_map_path)
    elif density_map_path.endswith('.csv'):
        # Dùng thư viện pandas để nạp nội dung csv nhanh chóng
        density_map = pd.read_csv(density_map_path, header=None).values
    else:
        raise ValueError("File Mật độ (Density map) bắt buộc phải là định dạng .npy hoặc .csv")

    # Tìm kiếm các đỉnh (peaks). 
    # Hệ số threshold_abs (ngưỡng cắt) có thể cần điều chỉnh tùy thuộc vào tham số 'normalizer' lúc sinh bản đồ
    # Tham số min_distance quy định khoảng cách vật lí tối thiểu giữa 2 người (tính bằng pixels)
    coordinates = peak_local_max(density_map, min_distance=3, threshold_abs=0.01)

    # Lặp qua tập tọa độ nhận được và vẽ các điểm tròn nhỏ màu đỏ (red circles)
    for p in coordinates:
        y, x = int(p[0]), int(p[1]) # skimage peak_local_max trả về dạng (y, x) chứ không phải (x, y)
        cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    
    # Lưu lại kết quả sau cùng
    cv2.imwrite(output_path, image)
    print(f"[Peaks] Đã lưu ảnh sau khi vẽ dự đoán tọa độ người tới: {output_path} (Đã phát hiện: {len(coordinates)} điểm)")

def overlay_heatmap(image_path, density_map_path, output_path):
    """
    Phủ một bản đồ phổ màu (color-mapped heatmap) - được trích xuất từ density map
    lên trên thẳng bức ảnh gốc bằng thuật toán Blending.
    """
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")

    # Đọc density map
    if density_map_path.endswith('.npy'):
        density_map = np.load(density_map_path)
    elif density_map_path.endswith('.csv'):
        density_map = pd.read_csv(density_map_path, header=None).values
    else:
        raise ValueError("File Mật độ (Density map) bắt buộc phải là định dạng .npy hoặc .csv")

    # Chuẩn hóa (Normalize) các giá trị trong density map về dạng [0, 1] 
    # để phục vụ cho việc gán chuẩn màu RGB
    den_min, den_max = density_map.min(), density_map.max()
    if den_max > den_min:
        density_map_normalized = (density_map - den_min) / (den_max - den_min)
    else:
        density_map_normalized = density_map
    
    # Ép kiểu dữ liệu về 8-bit (0-255) do OpenCV chỉ hỗ trợ màu số nguyên hệ 8 bit
    density_map_8bit = (density_map_normalized * 255).astype(np.uint8)

    # Chắc chắn rằng heatmap và ảnh gốc có cùng kích cỡ trước khi hợp nhất
    h, w = image.shape[:2]
    density_map_8bit = cv2.resize(density_map_8bit, (w, h))

    # Áp dụng bảng màu COLORMAP_JET lên màn ảnh mật độ để tạo thành màu hồng ngoại (đỏ - xanh)
    heatmap = cv2.applyColorMap(density_map_8bit, cv2.COLORMAP_JET)

    # Sử dụng hàm addWeighted để chồng Heatmap lên ảnh gốc. 
    # Trọng số 0.4 cho ảnh, 0.6 cho Heatmap. Tham số cuối (0) là bias (gamma).
    overlay = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
    
    # Save ảnh kết quả
    cv2.imwrite(output_path, overlay)
    print(f"[Heatmap] Đã lưu thành công ảnh Heatmap chồng lấp tại: {output_path}")

def main():
    # Khởi tạo bộ parse để bắt các tham số truyền vào từ Terminal
    parser = argparse.ArgumentParser(description="Kịch bản (Script) trực quan hóa: GT points, Peaks, và Heatmaps")
    parser.add_argument('--image', type=str, required=True, help="Đường dẫn đến file hình ảnh gốc (Vd: IMG_1.jpg)")
    parser.add_argument('--mat', type=str, help="Đường dẫn file đánh nhãn Ground Truth (.mat) để vẽ điểm gốc")
    parser.add_argument('--den', type=str, help="Đường dẫn kết quả mật độ do model sinh ra (.csv hoặc .npy) - Phục vụ tìm đỉnh và heatmap")
    parser.add_argument('--out_dir', type=str, default="results_vis", help="Thư mục sẽ được dùng để lưu kết quả trực quan hóa")
    
    args = parser.parse_args()

    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Lấy tên file gốc (trừ bỏ phần đuôi mở rộng) để tiện việc đặt tên các file mới
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # Nhánh 1: Vẽ điểm từ file GT nếu người dùng có cung cấp file .mat
    if args.mat:
        gt_out = os.path.join(args.out_dir, f"{base_name}_gt_points.jpg")
        draw_gt_points(args.image, args.mat, gt_out)
    
    # Nhánh 2: Vẽ dự đoán và heatmap dựa trên Mật độ (Density Map)
    if args.den:
        # Nhánh 2a: Tìm kiếm và vẽ local maxima/blobs
        peak_out = os.path.join(args.out_dir, f"{base_name}_pred_peaks.jpg")
        find_and_draw_peaks(args.image, args.den, peak_out)

        # Nhánh 2b: Vẽ heatmap
        heat_out = os.path.join(args.out_dir, f"{base_name}_heatmap.jpg")
        overlay_heatmap(args.image, args.den, heat_out)
        
if __name__ == '__main__':
    # Điểm entry của app khi chạy trực tiếp trên python environment
    main()
