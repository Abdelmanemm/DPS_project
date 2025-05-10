import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow


def grayscale(image: np.ndarray) -> np.ndarray:

   r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

   grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b

   return grayscale_img.astype(np.uint8)
