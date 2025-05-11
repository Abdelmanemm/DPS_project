import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies a 2D convolution to a grayscale or single-channel image.

    Args:
        image (np.ndarray): 2D input image array.
        kernel (np.ndarray): 2D kernel array.

    Returns:
        np.ndarray: Convolved image.
    """
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    convolved = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved[i, j] = np.sum(region * kernel)

    return convolved


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpens an image using the Laplacian kernel.

    Args:
        image (np.ndarray): Input image (2D grayscale or 3D RGB).

    Returns:
        np.ndarray: Sharpened image.
    """
    laplacian_kernel = np.array([
        [0, -1,  0],
        [-1,  4, -1],
        [0, -1,  0]
    ], dtype=np.float32)

    if image.ndim == 2:  # Grayscale image
        laplacian = convolve2d(image.astype(np.float32), laplacian_kernel)
        sharpened = image.astype(np.float32) + laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    elif image.ndim == 3:  # RGB image
        channels = []
        for c in range(image.shape[2]):
            laplacian = convolve2d(image[:, :, c].astype(np.float32), laplacian_kernel)
            sharpened = image[:, :, c].astype(np.float32) + laplacian
            channels.append(np.clip(sharpened, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=2)

    else:
        raise ValueError("Unsupported image dimensions. Expected 2D or 3D array.")
