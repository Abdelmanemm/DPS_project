import numpy as np
from typing import Tuple

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def convolve2d(image: np.ndarray, kernel: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    backend = cp if use_gpu and GPU_AVAILABLE else np
    kernel = backend.flipud(backend.fliplr(kernel))
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded_image = backend.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    convolved = backend.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved[i, j] = backend.sum(region * kernel)

    return backend.asnumpy(convolved) if use_gpu and GPU_AVAILABLE else convolved


def sharpen(image: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=np.float32)

    if image.ndim == 2:
        laplacian = convolve2d(image.astype(np.float32), laplacian_kernel, use_gpu)
        sharpened = image.astype(np.float32) + laplacian
        sharpened = np.clip(sharpened, 0, 255)
        return sharpened.astype(np.uint8)

    elif image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            laplacian = convolve2d(image[:, :, c].astype(np.float32), laplacian_kernel, use_gpu)
            sharpened = image[:, :, c].astype(np.float32) + laplacian
            sharpened = np.clip(sharpened, 0, 255)
            channels.append(sharpened.astype(np.uint8))
        return np.stack(channels, axis=2)

    else:
        raise ValueError("Unsupported image dimensions. Expected 2D or 3D array.")


def batch_sharpen(images: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    if images.ndim == 3:
        return np.stack([sharpen(img, use_gpu) for img in images], axis=0)
    elif images.ndim == 4:
        return np.stack([sharpen(img, use_gpu) for img in images], axis=0)
    else:
        raise ValueError("Expected batch input with shape (N, H, W) or (N, H, W, C)")


# CLI integration
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from imageio import imread, imwrite
    import os

    parser = argparse.ArgumentParser(description="Apply Laplacian sharpening to an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="sharpened_output.png", help="Path to save sharpened image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (if available)")
    parser.add_argument("--show", action="store_true", help="Display before and after images")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    img = imread(args.input)
    sharpened_img = sharpen(img, use_gpu=args.gpu)
    imwrite(args.output, sharpened_img)
    print(f"Sharpened image saved to {args.output}")

    if args.show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Sharpened")
        if sharpened_img.ndim == 2:
            plt.imshow(sharpened_img, cmap='gray')
        else:
            plt.imshow(sharpened_img)
        plt.axis('off')

        plt.show()

