# 🚗 Car Number Plate Detection

A Python-based computer vision project that detects and extracts text from vehicle license plates using **OpenCV** and **Tesseract OCR**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [License](#license)

---

## Overview

This project uses image processing techniques to detect license plates in vehicle images and extract the plate number using Optical Character Recognition (OCR). It processes an input image, identifies plate-like regions using edge detection and contour analysis, and reads the text from those regions.

---

## ✨ Features

- Detects number plate regions in vehicle images
- Extracts and prints text from detected plates using OCR
- Highlights detected plate contours on the original image
- Simple and beginner-friendly codebase

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| OpenCV (`cv2`) | Image processing & contour detection |
| Tesseract OCR | Text extraction from plate regions |
| pytesseract | Python wrapper for Tesseract |

---

## ✅ Prerequisites

Before running this project, make sure you have the following installed:

- Python 3.x
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Install from the official repo or via installer
  - **Windows:** Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Linux:** `sudo apt install tesseract-ocr`
  - **macOS:** `brew install tesseract`

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/car-numplate-detection.git
cd car-numplate-detection
```

2. **Install Python dependencies:**

```bash
pip install opencv-python pytesseract
```

3. **Configure Tesseract path** (Windows only):

Open `car_numplate_detection.py` and update this line to match your Tesseract installation path:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

On Linux/macOS, this line is usually not needed.

---

## ▶️ Usage

1. Place your vehicle image in the project directory.

2. Update the `image_path` variable in the script:

```python
image_path = 'your_image.jpg'
```

3. Run the script:

```bash
python car_numplate_detection.py
```

4. A window will display the image with detected plate contours highlighted in green. The extracted plate text will be printed in the terminal:

```
Plate 1: ABC-1234
```

---

## ⚙️ How It Works

1. **Load Image** — Reads the vehicle image using OpenCV.
2. **Grayscale Conversion** — Converts the image to grayscale for easier processing.
3. **Gaussian Blur** — Reduces noise to improve edge detection accuracy.
4. **Canny Edge Detection** — Detects edges in the blurred image.
5. **Contour Detection** — Finds all contours from the edges.
6. **Contour Filtering** — Filters out small contours (minimum area: 500 px²) to isolate plate-like regions.
7. **ROI Extraction** — Crops each potential plate region from the original image.
8. **OCR** — Uses Tesseract with `--psm 8` (single word mode) to read the plate text.

---

## 📁 Project Structure

```
car-numplate-detection/
│
├── car_numplate_detection.py   # Main detection script
├── README.md                   # Project documentation
└── sample_images/              # (Optional) Test images
```

---

## ⚠️ Limitations & Future Improvements

**Current Limitations:**
- Works best on clear, well-lit, front-facing plate images
- May detect false positives (non-plate regions with similar shape)
- OCR accuracy depends on image quality and plate font

**Planned Improvements:**
- [ ] Use a trained HAAR cascade or deep learning model for more accurate plate localization
- [ ] Add support for real-time webcam/video feed
- [ ] Improve OCR accuracy with image preprocessing (thresholding, deskewing)
- [ ] Support multiple image formats and batch processing
- [ ] Build a simple GUI for ease of use

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Made with ❤️ using Python, OpenCV, and Tesseract OCR
