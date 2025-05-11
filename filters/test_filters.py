'''Testing Filters to ensure correct functionality'''

import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image ,ImageOps , ImageFilter
from grayscale import grayscale
from resize import resize_nearest_neighbor , resize_bilinear
from horizontal_flip import horizontal_flip
from sharpen import sharpen
from edge_detection import sobel_edge_detection
from blur import apply_gaussian_blur
# Read img we will test on it
test_img  = imageio.imread("/mnt/d/Workspace/DPS_project/test/test_img.jpg")
print(f"Image loaded Succefully....")



def test_grayscale(img: np.ndarray) -> None:
    # Apply custom grayscale filter
    custom_gray = grayscale(img)  # Should return a 2D grayscale numpy array

    # Convert numpy image to PIL for comparison (PIL expects RGB)
    pil_gray = Image.fromarray(img).convert("L")  
    pil_gray_np = np.array(pil_gray)

    # Compute Mean Absolute Error between outputs
    mae = np.mean(np.abs(custom_gray.astype(np.float32) - pil_gray_np.astype(np.float32)))
    print(f"Mean Absolute Error (custom vs PIL): {mae:.2f}")

    # Display comparison
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(custom_gray, cmap="gray")
    axs[1].set_title("Implmented Grayscale")
    axs[1].axis("off")

    axs[2].imshow(pil_gray_np, cmap="gray")
    axs[2].set_title("PIL Grayscale")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def test_resize(img: np.ndarray, new_width: int, new_height: int):
    methods = ["nearest", "bilinear"]

    for method in methods:
        # Apply your custom resize
        if method == 'nearest':
            custom_resized = resize_nearest_neighbor(img, new_width, new_height)
        else:
            custom_resized = resize_bilinear(img, new_width, new_height)

        # Resize using PIL for comparison
        pil_img = Image.fromarray(img)
        pil_mode = Image.NEAREST if method == "nearest" else Image.BILINEAR
        pil_resized = pil_img.resize((new_height, new_width), resample=pil_mode)
        pil_resized_np = np.array(pil_resized)



        # Show results
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(custom_resized)
        axs[1].set_title(f"Custom Resize ({method})")
        axs[1].axis("off")

        axs[2].imshow(pil_resized_np)
        axs[2].set_title(f"PIL Resize ({method})")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

def test_horizontal_flip(img: np.ndarray):
    # Apply your flip
    print(img.shape)
    custom_flipped = horizontal_flip(img)

    # Use PIL to flip
    pil_img = Image.fromarray(img)
    pil_flipped = ImageOps.mirror(pil_img)
    pil_flipped_np = np.array(pil_flipped)

    # Compute Mean Absolute Error
    mae = np.mean(np.abs(custom_flipped.astype(np.float32) - pil_flipped_np.astype(np.float32)))
    print(f"[HORIZONTAL FLIP] MAE with PIL: {mae:.2f}")

    # Show results
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(custom_flipped)
    axs[1].set_title("Custom Horizontal Flip")
    axs[1].axis("off")

    axs[2].imshow(pil_flipped_np)
    axs[2].set_title("PIL Horizontal Flip")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def test_sharpen(test_img):

    # Our manual sharpen
    sharpened_manual = sharpen(test_img)

    # Reference sharpen from PIL
    pil_image = Image.fromarray(test_img)
    sharpened_pil = np.array(pil_image.filter(ImageFilter.SHARPEN))

    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_img)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(sharpened_manual)
    plt.title("Manual Sharpen (Laplacian)")

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_pil)
    plt.title("PIL Sharpen")

    plt.tight_layout()
    plt.show()

def test_edge_detection(image: np.ndarray):
    """
    Test custom edge detection filter by comparing it to PIL's FIND_EDGES filter.

    Args:
        image (np.ndarray): Input image as a NumPy array.
    """
    # Apply custom edge detection
    custom_edge = sobel_edge_detection(image)

    # Convert NumPy image to PIL Image for reference
    pil_image = Image.fromarray(image)
    pil_edge = pil_image.filter(ImageFilter.FIND_EDGES)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(custom_edge, cmap="gray")
    axes[1].set_title("Custom Edge Detection")
    axes[1].axis("off")

    axes[2].imshow(pil_edge, cmap="gray")
    axes[2].set_title("PIL FIND_EDGES")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def test_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 9.0):
    """
    Test Gaussian blur filter by comparing custom implementation to PIL's GaussianBlur.
    """

    # Custom Gaussian Blur
    custom_blur = apply_gaussian_blur(image, kernel_size, sigma)

    # PIL Gaussian Blur (convert to Image first)
    pil_img = Image.fromarray(image)
    pil_blur = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    pil_blur_np = np.array(pil_blur)

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(custom_blur)
    plt.title("Custom Gaussian Blur")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pil_blur_np)
    plt.title("PIL Gaussian Blur")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_grayscale(test_img)
    # test_resize(test_img,250,500)
    # test_horizontal_flip(test_img)
    # test_sharpen(test_img)
    # test_edge_detection(test_img)
    test_blur(test_img)

