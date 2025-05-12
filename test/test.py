import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import all your filters directly
from grayscale import grayscale
from edge_detection import sobel_edge_detection
from horizontal_flip import horizontal_flip
from resize import resize_bilinear
from sharpen import sharpen
from brightness import adjust_brightness
from contrast import adjust_contrast

def quick_test(image_path):
    """Runs all filters on test image and shows results"""
    # Load test image
    img = np.array(Image.open(image_path))
    print(f"Testing with image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    
    # Define test cases
    tests = [
        ("Grayscale", lambda x: grayscale(x)),
        ("Edge Detection", lambda x: sobel_edge_detection(x)),
        ("Horizontal Flip", lambda x: horizontal_flip(x)),
        ("Resize", lambda x: resize_bilinear(x, img.shape[0]//2, img.shape[1]//2)),
        ("Sharpen", lambda x: sharpen(x)),
        ("Brightness", lambda x: adjust_brightness(x, 1.5)),
        ("Contrast", lambda x: adjust_contrast(x, 50, 200))
    ]
    
    # Run tests
    for name, test_func in tests:
        try:
            print(f"\nTesting: {name}")
            result = test_func(img)
            
            # Show comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img if img.ndim == 3 else img, cmap='gray')
            plt.title("Original")
            
            plt.subplot(1, 2, 2)
            plt.imshow(result if result.ndim == 3 else result, cmap='gray')
            plt.title(name)
            
            plt.show()
            
            # Save test output
            output_path = f"test_{name.lower().replace(' ', '_')}.jpg"
            Image.fromarray(result).save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py <image_path>")
    else:
        quick_test(sys.argv[1])