# DPS_project
# 🔧 Project Tasks Overview

This project extends our image processing system by reimplementing all filters from scratch using DSP techniques and NumPy. It also adds simple audio filtering. The work is divided into structured phases with assigned team members and deliverables.

---

## 💼 Phase 1: Planning & Setup

### 📄 Project Refactoring Plan
**👤 Assigned to:** Khail  
- **Objective:** Analyze existing code structure, remove YOLO dependency, and propose modular improvements.  
- **Deliverables:** `refactor_plan.md`

### 📚 Research Image Filtering with DSP
**👤 Assigned to:** Salma & Radwa  
- **Objective:** Study DSP implementations for grayscale, Gaussian blur, edge detection, etc.  
- **Deliverables:** `filter_math_notes.pdf` (must include equations + steps)

---

## 🧠 Phase 2: Re-Implementation of Filters (from scratch using NumPy)

> Each filter must:
> - Be implemented manually (no PIL/ImageFilter)
> - Include inline DSP logic comments
> - Have a basic test using `pytest` or a test script

### 🎨 Grayscale Conversion from RGB
**👤 Assigned to:** Mahmoud  
- **File:** `filters/grayscale.py`

### 🌫 Gaussian Blur Filter (2D Convolution)
**👤 Assigned to:** Keto  
- **File:** `filters/gaussian_blur.py`

### 🧱 Edge Detection (Sobel or Prewitt)
**👤 Assigned to:** Belal  
- **File:** `filters/edge_detection.py`

### 🔁 Horizontal Flip (Manual Pixel Swap)
**👤 Assigned to:** Omar  
- **File:** `filters/horizontal_flip.py`

### 🖼 Resize with Nearest Neighbor / Bilinear
**👤 Assigned to:** Dalia  
- **File:** `filters/resize.py`

### 🌀 Image Contours (Gradient-based)
**👤 Assigned to:** Samy  
- **File:** `filters/contours.py`

### ✨ Image Sharpening (Laplacian Kernel)
**👤 Assigned to:** Ali  
- **File:** `filters/sharpen.py`

### 💡 Brightness Adjustment (Pixel Scaling)
**👤 Assigned to:** Lashen  
- **File:** `filters/brightness.py`

### 🏁 Contrast Adjustment (Histogram Stretching)
**👤 Assigned to:** Salma  
- **File:** `filters/contrast.py`

### 🎨 Color Saturation (HSV / HSL Adjustment)
**👤 Assigned to:** Radwa  
- **File:** `filters/saturation.py`

---

## 🔊 Phase 3: Audio Filter Implementation

### 🎧 Add Simple Audio Filters (low-pass & high-pass using FFT)
**👤 Assigned to:** Mahmoud & Dalia  
- **File:** `audio_filters.py`  
- **Bonus:** Add support for `soundfile` or `scipy.io.wavfile`

---

## 🧪 Phase 4: Integration & Testing

### 🔄 Update Menu and `preprocess_image()` to use new modules
**👤 Assigned to:** Khail & Keto  

### 🧪 Write Unit Tests for All Filters
**👤 Assigned to:** Samy & Ali  
- **File:** `tests/test_filters.py`

### 📘 User Manual & Final Report Writing
**👤 Assigned to:** Belal & Radwa  
- **Files:** `docs/final_report.pdf`, `README.md`
