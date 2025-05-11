import numpy as np
from sharpen import convolve2d

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.

    Parameters:
        size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Normalized Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies Gaussian blur to an image using 2D convolution.

    Parameters:
        image (np.ndarray): Grayscale or RGB image.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        np.ndarray: Blurred image.
    """
    kernel = gaussian_kernel(kernel_size, sigma)

    if image.ndim == 2:
        blurred = convolve2d(image, kernel)
    elif image.ndim == 3:
        blurred = np.stack([convolve2d(image[:, :, c], kernel) for c in range(image.shape[2])], axis=2)
    else:
        raise ValueError("Unsupported image format. Must be 2D or 3D NumPy array.")

    return np.clip(blurred, 0, 255).astype(np.uint8)
