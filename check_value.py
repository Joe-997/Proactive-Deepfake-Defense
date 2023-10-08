from PIL import Image
import numpy as np

def load_images_as_tensors(image_paths):
    tensors = []

    for path in image_paths:
        # 使用PIL打开图像
        img = Image.open(path)
        # 将图像转换为numpy数组
        img_array = np.array(img)
        img_array = img_array / 255.0
        # 将像素值从[0, 1]范围转换到[-1, 1]范围
        img_array = img_array * 2 - 1
        # 添加到张量列表中
        tensors.append(img_array)
    return tensors

if __name__ == "__main__":
    # 图片文件路径列表
    image_paths = ["C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/noise5.jpg",
                   "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/5attrs_adv/4perturb-image.jpg"]

    # 加载图片并输出张量形式
    tensors = load_images_as_tensors(image_paths)
    for i, tensor in enumerate(tensors):
        print(f"Image {i+1} tensor shape: {tensor.shape}")

