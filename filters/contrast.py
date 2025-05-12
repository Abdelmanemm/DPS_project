import numpy as np

def adjust_contrast(image: np.ndarray, min_out: int = 0, max_out: int = 255) -> np.ndarray:
    """Performs linear contrast stretching to [min_out, max_out].
    
    Args:
        image: Input image (2D grayscale or 3D RGB).
        min_out: Minimum output value.
        max_out: Maximum output value.
    
    Returns:
        np.ndarray: Contrast-stretched image.
    """
    if image.ndim == 3:  # RGB
        return np.stack([
            adjust_contrast(image[..., c], min_out, max_out)
            for c in range(image.shape[2])
        ], axis=2)
    
    min_in, max_in = image.min(), image.max()
    if min_in == max_in:  # Avoid division by zero
        return image
    
    stretched = (image - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out
    return np.clip(stretched, min_out, max_out).astype(np.uint8)