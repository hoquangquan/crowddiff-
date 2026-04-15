import torch as th
import numpy as np

import cv2
import os

from einops import rearrange
from PIL import Image

from matplotlib import pyplot as plt
from skimage.feature import peak_local_max


class DataParameter():
    """
    Lớp đóng gói trạng thái và cấu hình (parameter state) cho mỗi ảnh (batch) 
    để hỗ trợ thuật toán sliding window nội suy, lưu tạm thời kết quả MAE lấy mẫu
    và cập nhật liên tục tiến trình đo đạc lượng đám đông qua nhiều vòng lặp (cycles).
    """
    def __init__(self, model_kwargs, args) -> None:

        self.name = model_kwargs['name'][0].split('-')[0]
        self.density = model_kwargs['high_res'].squeeze().numpy() # shape: (c,h,w)
        self.image = (Denormalize(model_kwargs['low_res'].squeeze().numpy())*255).astype(np.uint8).transpose(1,2,0)
        
        # denormalize the density map
        # self.density = Denormalize(self.density, normalizer=args.normalizer)

        # create image crops and get the count of each crop
        create_crops(model_kwargs, args)
        # create_overlapping_crops(model_kwargs, args)
        model_kwargs['low_res'] = model_kwargs['low_res']

        # operational parameters
        self.dims = np.asarray(model_kwargs['dims'])
        self.order = model_kwargs['order']
        self.resample = True
        self.cycles = 0
        self.image_size = args.large_size
        self.total_samples = args.per_samples
        # self.x_pos = model_kwargs['x_pos']
        # self.y_pos = model_kwargs['y_pos']

        # result parameters
        self.crowd_count = model_kwargs['crowd_count']
        self.mae = np.full(model_kwargs['high_res'].size(0), np.inf)
        self.result = np.zeros(model_kwargs['high_res'].size()) 
        self.result = np.mean(self.result, axis=1, keepdims=True)       

        # remove unnecessary keywords
        update_keywords(model_kwargs)


    def update_cycle(self):
        self.cycles += 1


    def evaluate(self, samples, model_kwargs):
        """
        Tiến hành xử lý đánh giá hiệu năng (tính sai phân count)
        Duy trì nếu MAE của lượt sinh mới nhỏ hơn vòng trước.
        """
        samples = samples.cpu().numpy()
        
        for index in range(self.order.size):
            if index >= len(samples):
                break
            p_result, p_mae = self.evaluate_sample(samples[index], index)
            if np.abs(p_mae) < np.abs(self.mae[self.order[index]]):
                self.result[self.order[index]] = p_result
                self.mae[self.order[index]] = p_mae

        indices = np.where(np.abs(self.mae[self.order])>0)
        self.order = self.order[indices]
        model_kwargs['low_res'] = model_kwargs['low_res'][indices]

        pred_count = self.get_total_count()

        length = len(self.order)!= 0
        cycles = self.cycles < self.total_samples
        error = np.sum(np.abs(self.mae[self.order]))>2

        self.resample = length and cycles and error
        
        print(f'mae: {self.mae}')
        progress = ' '.join([f'name: {self.name}',
                             f'cum mae: {np.sum(np.abs(self.mae[self.order]))}',
                             f'comb mae: {np.abs(pred_count-np.sum(self.crowd_count))}',
                             f'cycle:{self.cycles}'
                             ])
        # print(f'name: {self.name}, cum mae: {np.sum(np.abs(self.mae[self.order]))} \
        # comb mae: {np.abs(pred_count-np.sum(self.crowd_count))} cycle:{self.cycles}')
        print(progress)
    
    def get_total_count(self):
        
        image = self.combine_crops(self.result)
        pred_count = self.get_circle_count(image.astype(np.uint8))

        return pred_count


    def evaluate_sample(self, sample, index):
        
        sample = sample.squeeze()
        sample = (sample+1)
        sample = sample[0]
        sample = (sample/(sample.max()+1e-8))*255
        sample = sample.clip(0,255).astype(np.uint8)

        sample = remove_background(sample, count=200)
        
        pred_count = self.get_circle_count(sample, draw=False)

        return sample, pred_count-self.crowd_count[self.order[index]]


    def get_circle_count(self, image, threshold=0, draw=False, name=None):
        """
        Kết hợp các filter làm mờ, morphology và contours (trong OpenCV) để
        phân tách các điểm tập trung mật độ thành từng vòng tròn, qua đó đếm số lượng đám đông.
        """
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
            contoursImg = np.zeros_like(morphImg)
            contoursImg = np.repeat(contoursImg[:,:,np.newaxis],3,-1)
            for point in contours:
                x,y = point.squeeze().mean(0)
                if x==127.5 and y==127.5:
                    continue
                cv2.circle(contoursImg, (int(x),int(y)), radius=3, thickness=-1, color=(255,255,255))
            threshedImg = np.repeat(threshedImg[:,:,np.newaxis], 3,-1)
            morphImg = np.repeat(morphImg[:,:,np.newaxis], 3,-1)
            image = np.concatenate([contoursImg, threshedImg, morphImg], axis=1)
            cv2.imwrite(f'experiments/target_test/{name}_image.jpg', image)
        return max(len(contours)-1,0) # remove the boarder


    def combine_crops(self, crops):

        crops = th.tensor(crops)
        p1, p2 = 1+(self.dims[0]-1)//self.image_size, 1+(self.dims[1]-1)//self.image_size
        crops = rearrange(crops, '(p1 p2) c h w -> (p1 h) (p2 w) c',p1=p1, p2=p2)
        crops = crops.squeeze().numpy()

        start_h, start_w = (crops.shape[0]-self.dims[0])//2, (crops.shape[1]-self.dims[1])//2
        end_h, end_w = start_h+self.dims[0], start_w+self.dims[1]

        image = crops[start_h:end_h, start_w:end_w]
        
        return image


    def combine_overlapping_crops(self, crops):
        
        # if len(crops[0].shape) == 4:
        image = th.zeros((crops.shape[1],self.dims[0],self.dims[1]))
        # else:
            # image = th.zeros((1,self.dims[0],self.dims[1]))
        crops = crops.cpu()

        mask = th.zeros(image.shape)

        count = 0
        for i in self.y_pos:
            for j in self.x_pos:
                if count == crops.shape[0]:
                    image= image / (mask+1e-8)
                    return image
                image[:,i:i+self.image_size,j:j+self.image_size] = crops[count] + image[:,i:i+self.image_size,j:j+self.image_size]

                mask[:,i:i+self.image_size,j:j+self.image_size] = mask[:,i:i+self.image_size,j:j+self.image_size] + \
                    th.ones((crops.shape[1], self.image_size, self.image_size))
                count += 1
        image = image / mask

        return image
    

    def save_results(self, args):

        pred_count = self.get_total_count()
        gt_count = np.sum(self.crowd_count)

        comb_mae = np.abs(pred_count-gt_count)
        cum_mae = np.sum(np.abs(self.mae[self.order]))
        
        if comb_mae > cum_mae:
            pred_count = gt_count + np.sum(self.mae[self.order])
        density_map = self.combine_crops(self.result)
        
        # 1. Chuẩn bị ảnh gốc
        h, w = self.image.shape[:2]
        image_bgr = self.image.astype(np.uint8)[:,:,::-1]
        
        # --- PHASE 1: Vẽ chấm xanh (Green points / Peaks) ---
        image_peaks = image_bgr.copy()
        # min_distance 3, threshold 0.01
        coordinates = peak_local_max(density_map, min_distance=3, threshold_abs=50)
        for p in coordinates:
            y, x = int(p[0]), int(p[1])
            cv2.circle(image_peaks, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # --- PHASE 2: Vẽ Heatmap ---
        den_min, den_max = density_map.min(), density_map.max()
        if den_max > den_min:
            density_map_normalized = (density_map - den_min) / (den_max - den_min)
        else:
            density_map_normalized = density_map
            
        density_map_8bit = (density_map_normalized * 255).astype(np.uint8)
        density_map_8bit = cv2.resize(density_map_8bit, (w, h))

        heatmap_bgr = cv2.applyColorMap(density_map_8bit, cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(image_bgr, 0.4, heatmap_bgr, 0.6, 0)

        # 3. Nối hai hình ảnh lại thành 1 để tiện xem cả hai kết quả (Trái: Chấm xanh, Phải: Heatmap)
        combined_out = np.concatenate([image_peaks, overlay_bgr], axis=1)

        # Lưu ảnh cuối cùng
        cv2.imwrite(os.path.join(args.log_dir, f'{self.name} {int(pred_count)} {int(gt_count)}.jpg'), combined_out)



def remove_background(image, count=None):
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

    count = count_colors(image) if count is None else count
    image = image*(image>count)

    return image


def update_keywords(model_kwargs):
    image = model_kwargs['low_res']
    keys = list(model_kwargs.keys())
    for key in keys:
        del model_kwargs[key]
    model_kwargs['low_res'] = image


def Denormalize(image, normalizer=1):
    """Apply the inverse normalization to the image
    inputs: image to denormalize and normalizing constant
    output: image with values between 0 and 1
    """
    image = (image+1)*normalizer*0.5
    return image


def create_crops(model_kwargs, args):
    """Create image crops from the crowd dataset
    inputs: crowd image, density map
    outputs: model_kwargs and crowd count
    """
    
    image = model_kwargs['low_res']
    density = model_kwargs['high_res']

    model_kwargs['dims'] = density.shape[-2:]

    # create a padded image
    image = create_padded_image(image, args.large_size)
    density = create_padded_image(density, args.large_size)

    model_kwargs['low_res'] = image
    model_kwargs['high_res'] = density

    # print(model_kwargs['high_res'].shape)
    # print(th.sum(model_kwargs['high_res'][:,0]), th.sum(model_kwargs['high_res'])/3)
    # model_kwargs['crowd_count'] = th.sum((model_kwargs['high_res']+1)*0.5*args.normalizer, dim=(1,2,3)).cpu().numpy()
    model_kwargs['crowd_count'] = np.stack([crop[0].sum().round().item() for crop in model_kwargs['high_res']])
    model_kwargs['order'] = np.arange(model_kwargs['low_res'].size(0))
    
    organize_crops(model_kwargs)


def create_padded_image(image, image_size):

    _, c, h, w = image.shape
    p1, p2 = (h-1+image_size)//image_size, (w-1+image_size)//image_size
    pad_image = th.full((1,c,p1*image_size, p2*image_size),0, dtype=image.dtype)

    start_h, start_w = (p1*image_size-h)//2, (p2*image_size-w)//2
    end_h, end_w = h+start_h, w+start_w

    pad_image[:,:,start_h:end_h, start_w:end_w] = image
    pad_image = rearrange(pad_image, 'n c (p1 h) (p2 w) -> (n p1 p2) c h w', p1=p1, p2=p2)

    return pad_image


def organize_crops(model_kwargs):
    indices = np.where(model_kwargs['crowd_count']>=0)
    model_kwargs['order'] = model_kwargs['order'][indices]
    model_kwargs['low_res'] = model_kwargs['low_res'][indices]


def create_overlapping_crops(model_kwargs, args):
    """
    Create overlapping image crops from the crowd image
    inputs: model_kwargs, arguments

    outputs: model_kwargs and crowd count
    """
    
    image = model_kwargs['low_res']
    density = model_kwargs['high_res']

    model_kwargs['dims'] = density.shape[-2:]
    
    X_points = start_points(size=model_kwargs['dims'][1],
                            split_size=args.large_size,
                            overlap=args.overlap
                            )
    Y_points = start_points(size=model_kwargs['dims'][0],
                            split_size=args.large_size,
                            overlap=args.overlap
                            )

    image = arrange_crops(image=image, 
                          x_start=X_points, y_start=Y_points,
                          crop_size=args.large_size
                          )
    density = arrange_crops(image=density, 
                            x_start=X_points, y_start=Y_points,
                            crop_size=args.large_size
                            )

    model_kwargs['low_res'] = image
    model_kwargs['high_res'] = density

    model_kwargs['crowd_count'] = th.sum((model_kwargs['high_res']+1)*0.5*args.normalizer, dim=(1,2,3)).cpu().numpy()
    model_kwargs['order'] = np.arange(model_kwargs['low_res'].size(0))

    model_kwargs['x_pos'] = X_points
    model_kwargs['y_pos'] = Y_points


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def arrange_crops(image, x_start, y_start, crop_size):
    crops = []
    for i in y_start:
        for j in x_start:
            split = image[:,:,i:i+crop_size, j:j+crop_size]
            crops.append(split)
    
    crops = th.stack(crops)
    crops = rearrange(crops, 'n b c h w-> (n b) c h w')
    return crops