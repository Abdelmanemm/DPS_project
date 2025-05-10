
import numpy as np
import cv2
import os
from grayscale import grayscale

def test_image_grayscale_conversion():
    # Create a dummy RGB image (10x10) with color pattern
    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    test_img[:] = [60, 120, 200]  # Sample RGB color

    test_img_path = "test_rgb_image.png"
    gray_img_path = "test_grayscale_image.png"

    # Save the dummy RGB image
    cv2.imwrite(test_img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

    # Apply grayscale conversion
    gray_image = grayscale(test_img)

    # Save the grayscale image for visual inspection
    cv2.imwrite(gray_img_path, gray_image)

    # Check if grayscale image is 2D and saved
    assert os.path.exists(gray_img_path), "Grayscale image file not saved."
    assert len(gray_image.shape) == 2, "Output image is not 2D grayscale."

    print("test_image_grayscale_conversion passed and grayscale image is saved.")

if __name__ == "__main__":
    test_image_grayscale_conversion()
