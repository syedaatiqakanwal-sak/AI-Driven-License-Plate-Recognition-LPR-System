# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:13:19 2023

@author: PMYLS
"""
import cv2
import pytesseract

# Set the path to the Tesseract OCR executable (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image using OpenCV
image_path = 'C:/Users/PMLS/Desktop/numpl.jfif'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and help with contour detection
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Use the Canny edge detector to find edges in the image
edges = cv2.Canny(blur, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to find potential plates
min_plate_area = 500
plate_contours = [contour for contour in contours if cv2.contourArea(contour) > min_plate_area]

# Draw the contours on the original image
cv2.drawContours(image, plate_contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract text from the detected plates using Tesseract OCR
for i, plate_contour in enumerate(plate_contours):
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_roi = image[y:y+h, x:x+w]

    # Convert the plate region to grayscale
    plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text from the plate
    text = pytesseract.image_to_string(plate_gray, config='--psm 8')

    print(f"Plate {i + 1}: {text.strip()}")

