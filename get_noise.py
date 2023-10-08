from PIL import Image
import os
import torch
from torchvision import transforms


def compute_noise_image(image_paths):
    noise_images = []

    for i in range(len(image_paths) - 1, 0, -1):
        img1_path = image_paths[i]
        img2_path = image_paths[i - 1]

        transform = transforms.ToTensor()
        img1 = transform(Image.open(img1_path))
        img2 = transform(Image.open(img2_path))

        # 计算噪声图像
        noise_image = torch.abs(img1 - img2)

        # 将噪声图像转换为PIL Image
        noise_image = transforms.ToPILImage()(noise_image)

        # 添加到噪声图像列表
        noise_images.append(noise_image)

    return noise_images


def save_noise_images(noise_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, noise_image in enumerate(noise_images):
        save_path = os.path.join(output_dir, f'noise{i + 1}.jpg')
        noise_image.save(save_path)
        print(f"Noise image {i + 1} saved at: {save_path}")


if __name__ == "__main__":
    # 请将image_paths替换为你实际的图片文件路径列表，列表中的图片应按照5到0的顺序排列
    image_paths = [
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/5real-image-256.jpg",
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/4real-image-256.jpg",
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/3real-image-256.jpg",
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/2real-image-256.jpg",
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/1real-image-256.jpg",
        "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/0real-image-256.jpg"]

    # 计算噪声图像
    noise_images = compute_noise_image(image_paths)

    # 保存噪声图像
    output_dir = "C:/projects/custmize_disrupt_stargan/stargan_celeba_256/test_space/256_x_real-0-5/"  # 将 "噪声图像保存路径" 替换为你实际的保存路径
    save_noise_images(noise_images, output_dir)
