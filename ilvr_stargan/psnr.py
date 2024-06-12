import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float32
import matplotlib.pyplot as plt


def psnr(original_img, generated_img, max_pixel_value=1.0):
    mse = F.mse_loss(original_img, generated_img)
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value.item()



count_value = 0
for i in range(1, 11):
    filename = f"{str(i).zfill(6)}.jpg"
    org = torch.from_numpy(np.array(Image.open(f"results/{i}-ilvr-image-sample.jpg").convert('RGB'))).permute(2, 0, 1).float() / 255.0
    #org = torch.from_numpy(np.array(Image.open(f"data/celeba/images/" + filename).convert('RGB'))).permute(2, 0, 1).float() / 255.0
    img_org = org.unsqueeze(0)

    sample = torch.from_numpy(np.array(Image.open(f"results/{i}-org9-sample.jpg").convert('RGB'))).permute(2, 0,
                                                                                                                 1).float() / 255.0
    #sample = torch.from_numpy(np.array(Image.open(f"results/{i}-real-image-sample.jpg").convert('RGB'))).permute(2, 0,1).float() / 255.0
    #sample = torch.from_numpy(np.array(Image.open(f"results/{i}-ilvr_strong_sample.jpg").convert('RGB'))).permute(2, 0, 1).float() / 255.0
    img_sample = sample.unsqueeze(0)

    result_img = psnr(img_org, img_sample)
    if result_img == float('inf'):
        result_img = 100

    print(f"PSNR_sample{i}:", result_img)
    count_value += result_img

print(f"PSNR_average:", count_value / 10)
"""


def compare_images_ssim(image_path1, image_path2):
    # 读取两张图片
    img1 = io.imread(image_path1)
    img2 = io.imread(image_path2)

    # 如果图片是彩色的，转换为灰度图
    if len(img1.shape) > 2:
        img1 = rgb2gray(img1)
    if len(img2.shape) > 2:
        img2 = rgb2gray(img2)

    # 转换图像为浮点型
    img1 = img_as_float32(img1)
    img2 = img_as_float32(img2)

    # 计算SSIM
    ssim_value, diff = ssim(img1, img2, full=True, data_range=img1.max() - img1.min())

    # diff是差异图像，将其显示出来（如果需要）
    #plt.imshow(diff, cmap='gray')
    #plt.title(f'SSIM: {ssim_value:.3f}')
    #plt.colorbar()
    #plt.show()
    
    return ssim_value


count_value = 0
for i in range(1, 11):
    filename = f"{str(i).zfill(6)}.jpg"
    image1_path = f'data/celeba/images/' + filename
    image2_path = f'results/{i}-ilvr_strong_result.jpg'
    #image2_path = f'results/{i}-real-image-sample.jpg'   # 与原图样本对比
    #image1_path = f'results/{i}-ilvr-image.jpg'          # 与ilvr纯净版对比
    #image2_path = f'results/{i}-ilvr-image-sample.jpg'    # 与ilvr_sample对比
    #image2_path = f'results/{i}-ilvr-black.jpg'


    ssim_index = compare_images_ssim(image1_path, image2_path)
    count_value += ssim_index
    print(ssim_index)


print(f"SSIM:", count_value / 10)
"""