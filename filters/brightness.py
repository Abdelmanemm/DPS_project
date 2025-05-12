import numpy as np

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjusts image brightness by scaling pixel values.
    
    Args:
        image: Input image (2D or 3D array).
        factor: Brightness multiplier (e.g., 1.0 = no change).
    
    Returns:
        np.ndarray: Brightness-adjusted image clipped to [0, 255].
    """
    adjusted = image.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)