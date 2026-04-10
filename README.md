AI-Driven License Plate Recognition (LPR) System
A specialized Computer Vision project focusing on automated vehicle identification through edge detection and Optical Character Recognition (OCR). This project reflects my core work in Deep Learning and Medical Imaging by applying similar image processing techniques (Gaussian blurring, contouring, and grayscale normalization) to real-world objects.

🚀 Overview
This system utilizes OpenCV and Tesseract OCR to detect, isolate, and read text from vehicle license plates. By applying a multi-stage image processing pipeline, the system filters out environmental noise to accurately identify plate contours.

🛠️ Technical Stack
Language: Python

Computer Vision: OpenCV (cv2)

OCR Engine: Tesseract OCR

Image Processing: * Grayscale Conversion

Gaussian Blur (Noise Reduction)

Canny Edge Detection

Contour Filtering & ROI (Region of Interest) Extraction

📋 Pipeline Architecture
The script follows a structured Computer Vision workflow:

Preprocessing: Convert input images to grayscale and apply Gaussian filtering to minimize background noise.

Edge Detection: Implement the Canny algorithm to identify the high-contrast boundaries of the plate.

Contour Analysis: Filter detected contours by area to differentiate the license plate from other vehicular features.

OCR Extraction: Crop the Region of Interest (ROI) and utilize Tesseract’s Page Segmentation Mode (--psm 8) optimized for single-word/short-string recognition.

⚙️ Installation & Usage
Install Tesseract OCR: Ensure Tesseract is installed on your local machine and update the path in the script:

Python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Install Dependencies:

Bash
pip install opencv-python pytesseract
Run the Script:

Bash
python "car numplate detection.py"
📈 Future Enhancements
Deep Learning Integration: Transitioning from Canny edge detection to a YOLOv8 or SSD model for more robust detection in low-light/high-speed environments.

Medical Imaging Parallel: Applying these segmentation techniques to diagnostic imaging for automated feature detection in Medical Imaging.

Developed by Syeda Atiqa Kanwal
