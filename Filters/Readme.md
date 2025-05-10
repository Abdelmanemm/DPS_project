# 🧪 Filters Directory

This folder contains all the custom image filter implementations written from scratch using **NumPy**. The goal is to understand and implement basic digital signal processing (DSP) concepts manually — without relying on high-level image libraries like PIL or OpenCV.

## 📌 Guidelines

- **No external image filter libraries allowed.**
- Use **NumPy** for all operations (array manipulation, math).
- Include **clear inline comments** explaining the DSP logic.
- Each script should be named according to its function (e.g., `grayscale.py`, `gaussian_blur.py`).
- Each filter should include:
  - A function that applies the filter to a NumPy image array.
  - A simple test or usage example at the bottom or in the `tests/` folder.
  - Input: `numpy.ndarray` (RGB or grayscale image).
  - Output: `numpy.ndarray` (processed image).

 ## 🧑‍💻 Coding Standard
- Use clear and descriptive function and variable names.
- Make sure your code is clean 
- Follow this code example as a reference
  ```python
  def grayscale(image: np.ndarray) -> np.ndarray:
      """Convert an RGB image to grayscale.

      Args:
          image (np.ndarray): RGB input image.

      Returns:
          np.ndarray: Grayscale image.
      """

## 🗂 Implemented Filters

| File                     | Description                       | Assigned Developer |
|--------------------------|-----------------------------------|---------------------|
| `grayscale.py`           | Convert RGB image to grayscale    | Mahmoud             |
| `gaussian_blur.py`       | Apply Gaussian blur (2D conv)     | Keto                |
| `edge_detection.py`      | Detect edges (Sobel/Prewitt)      | Belal               |
| `horizontal_flip.py`     | Flip image horizontally           | Omar                |
| `resize.py`              | Resize image (NN or Bilinear)     | Dalia               |
| `contours.py`            | Extract contours (gradient-based) | Samy                |
| `sharpen.py`             | Sharpen image (Laplacian kernel)  | Ali                 |
| `brightness.py`          | Adjust brightness (scaling)       | Lashen              |
| `contrast.py`            | Adjust contrast (histogram)       | Salma               |
| `saturation.py`          | Adjust saturation (HSV/HSL)       | Radwa               |

## 📁 File Structure

```
  filters/
│
├── grayscale.py
├── gaussian_blur.py
├── edge_detection.py
├── horizontal_flip.py
├── resize.py
├── contours.py
├── sharpen.py
├── brightness.py
├── contrast.py
└── saturation.py
```

## ✅ Testing

Each filter should be tested independently using `pytest` or a simple script. Make sure your code handles edge cases and invalid inputs gracefully.

---

Happy coding! 🧠🎨
