import numpy as np

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a grayscale image using the luminance method.

    Parameters:
        image (np.ndarray): A 3D NumPy array representing an RGB image. 
                            Shape must be (height, width, 3), where the 
                            third dimension represents the Red, Green, and Blue channels.

    Returns:
        np.ndarray: A 2D NumPy array representing the grayscale image, 
                    with pixel values in the range [0, 255] and dtype uint8.
    """

    # Extract individual Red, Green, and Blue channels from the input image
    r = image[:, :, 0]  # Red channel
    g = image[:, :, 1]  # Green channel
    b = image[:, :, 2]  # Blue channel

    # Apply the standard luminance formula for grayscale conversion
    # This formula accounts for human perception, giving more weight to green and less to blue
    grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # Convert the resulting floating-point array to unsigned 8-bit integer format
    # This is the standard format for grayscale images
    return grayscale_img.astype(np.uint8)

