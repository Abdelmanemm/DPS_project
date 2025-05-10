
import numpy as np

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Applies a horizontal flip to a grayscale image using manual pixel swapping.

    Parameters:
        image (np.ndarray): 2D array representing the grayscale image.

    Returns:
        np.ndarray: Horizontally flipped image.
    """

    # Create a copy of the image to store the result
    flipped_image = np.copy(image)

    height, width = image.shape

    # DSP Logic: swap pixels symmetrically around the vertical center
    for row in range(height):
        for col in range(width // 2):
            # Swap pixel at (row, col) with (row, width - col - 1)
            temp = flipped_image[row, col]
            flipped_image[row, col] = flipped_image[row, width - col - 1]
            flipped_image[row, width - col - 1] = temp

    return flipped_image
