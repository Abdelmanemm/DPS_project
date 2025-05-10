
import numpy as np
import cv2
import matplotlib.pyplot as plt
from filters.horizontal_flip import horizontal_flip

def load_and_prepare_image(path):
    # Read image using OpenCV
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Couldn't load image at path: {path}")
    return image

def show_images(original, flipped):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Horizontally Flipped")
    plt.imshow(flipped, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # 👇 Replace this with your image path
    image_path = "your_image.jpg"

    # Load and process image
    original = load_and_prepare_image(image_path)
    flipped = horizontal_flip(original)

    # Show the result
    show_images(original, flipped)

    # Optional: Save flipped image
    cv2.imwrite("flipped_output.jpg", flipped)
    print("✅ Image flipped and saved as 'flipped_output.jpg'.")

if __name__ == "__main__":
    main()
