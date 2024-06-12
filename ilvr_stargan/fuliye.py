import torch
import torch.fft


def gaussian_kernel(size, sigma):
    grid = torch.meshgrid([torch.arange(size, dtype=torch.float32) - (size - 1) / 2 for _ in range(2)])
    kernel = torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def apply_inverse_filter(image, kernel, device):
    # furier transform
    img_fft = torch.fft.fft2(image).to(device)
    kernel_fft = torch.fft.fft2(kernel, s=image.shape[-2:]).to(device)

    # calculate the inverse filter
    kernel_fft_inverse = 1 / (kernel_fft + 1e-10)  # add a small value to avoid division by zero
    """
    # change the intensity of the inverse filter if needed
    magnitude = torch.abs(kernel_fft) + 1e-10  
    phase = torch.angle(kernel_fft)
    magnitude_inv = torch.clamp(1 / magnitude, max=1.0)
    kernel_fft_inverse = magnitude_inv * torch.exp(1j * phase)
    """
    # apply the inverse filter
    restored_fft = img_fft * kernel_fft_inverse
    restored_image = torch.fft.ifft2(restored_fft).real

    return restored_image

