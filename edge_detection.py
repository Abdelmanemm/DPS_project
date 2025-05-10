import numpy as np
def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Gy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    height, width = image.shape

    edge_image = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            region = padded_image[i-1:i+2, j-1:j+2] 
            gx = np.sum(region * Gx)  
            gy = np.sum(region * Gy)  
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            edge_image[i-1, j-1] = edge_magnitude

    edge_image = (edge_image / edge_image.max()) * 255
    return edge_image.astype(np.uint8)
