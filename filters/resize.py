import numpy as np

# Nearest Neighbor
def resize_nearest_neighbor(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resize image using nearest neighbor interpolation.

    Args:
        image (np.ndarray): Input image array (H x W x C).
        new_height (int): Desired height.
        new_width (int): Desired width.

    Returns:
        np.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    # Create empty output image
    resized = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # Compute scale
    row_scale = height / new_height
    col_scale = width / new_width

    for i in range(new_height):
        for j in range(new_width):
            # Find nearest neighbor in original image
            src_row = int(i * row_scale)
            src_col = int(j * col_scale)
            resized[i, j] = image[src_row, src_col] if channels > 1 else image[src_row, src_col]

    return resized.squeeze()


# Bilinear
def resize_bilinear(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resize image using bilinear interpolation.

    Args:
        image (np.ndarray): Input image array (H x W x C).
        new_height (int): Desired height.
        new_width (int): Desired width.

    Returns:
        np.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    resized = np.zeros((new_height, new_width, channels), dtype=np.float32)

    row_scale = height / new_height
    col_scale = width / new_width

    for i in range(new_height):
        for j in range(new_width):
            x = i * row_scale
            y = j * col_scale

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, height - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, width - 1)

            dx = x - x0
            dy = y - y0

            for c in range(channels):
                top = (1 - dy) * image[x0, y0, c] + dy * image[x0, y1, c]
                bottom = (1 - dy) * image[x1, y0, c] + dy * image[x1, y1, c]
                resized[i, j, c] = (1 - dx) * top + dx * bottom

    return np.clip(resized, 0, 255).astype(np.uint8).squeeze()





