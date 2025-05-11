import numpy as np

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Applies a horizontal flip to a grayscale or RGB image using manual pixel swapping.

    Parameters:
        image (np.ndarray): 2D (grayscale) or 3D (RGB) numpy array image.

    Returns:
        np.ndarray: Horizontally flipped image.
    """
    flipped_image = np.copy(image)

    if image.ndim == 2:  # Grayscale image
        height, width = image.shape
        for row in range(height):
            for col in range(width // 2):
                flipped_image[row, col], flipped_image[row, width - col - 1] = (
                    flipped_image[row, width - col - 1],
                    flipped_image[row, col],
                )

    elif image.ndim == 3:  # RGB or multi-channel image
        height, width, channels = image.shape
        for row in range(height):
            for col in range(width // 2):
                for ch in range(channels):
                    flipped_image[row, col, ch], flipped_image[row, width - col - 1, ch] = (
                        flipped_image[row, width - col - 1, ch],
                        flipped_image[row, col, ch],
                    )

    else:
        raise ValueError("Unsupported image shape: must be 2D or 3D numpy array.")

    return flipped_image
