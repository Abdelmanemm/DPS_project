import numpy as np
import cv2
import os
from grayscale import grayscale

def test_uploaded_image_grayscale_conversion(image_path: str):
    # Load the uploaded image
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

    # Convert BGR (OpenCV default) to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Apply grayscale conversion
    gray_image = grayscale(rgb_image)

    # Save the grayscale image for inspection
    gray_img_path = "uploaded_grayscale_output.png"
    cv2.imwrite(gray_img_path, gray_image)

    # Validate the output
    assert os.path.exists(gray_img_path), "Grayscale image file not saved."
    assert len(gray_image.shape) == 2, "Output image is not 2D grayscale."

    print("✅ Uploaded image was successfully converted to grayscale.")
    plt.imshow(gray_image, cmap='gray') 
    plt.title('Grayscale Image')
    plt.show()

if __name__ == "__main__":
    # Replace 'your_image.jpg' with the actual image file path
    test_uploaded_image_grayscale_conversion("your_image.jpg")

