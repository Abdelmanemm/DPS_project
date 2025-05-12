import numpy as np
from grayscale import grayscale
def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Applies Sobel edge detection to a grayscale image.

    Args:
        image (np.ndarray): 2D grayscale image.

    Returns:
        np.ndarray: Edge-detected image as 8-bit grayscale.
    """
    if  image.ndim !=2:
        image = grayscale(image)

    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    image = image.astype(np.float32)
    height, width = image.shape
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')  

    edges = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            region = padded[i:i+3, j:j+3]
            gx = np.sum(region * Gx)
            gy = np.sum(region * Gy)
            edges[i, j] = np.sqrt(gx**2 + gy**2)

    edges = np.clip((edges / edges.max()) * 255, 0, 255)
    return edges.astype(np.uint8)
