"""
Project 120. Text extraction from images
Description:
Text extraction from images involves identifying and extracting readable text from natural scenes, scanned documents, screenshots, or photos. In this project, we use OpenCV for image preprocessing and Tesseract OCR to extract textual content from an image file.

About:
📄 What This Project Demonstrates:
Detects and extracts text from images using OCR

Uses image preprocessing (thresholding, noise reduction) to enhance accuracy

Ideal for scene text, business cards, IDs, signboards, etc.
"""

# Install if not already: pip install pytesseract opencv-python pillow
# Install Tesseract OCR binary from https://github.com/tesseract-ocr/tesseract
 
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
 
# Load the image
image_path = "scene_text.jpg"  # Replace with your own image path
image = cv2.imread(image_path)
 
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Optional: apply thresholding to improve OCR
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
 
# Optional: noise removal
gray = cv2.medianBlur(gray, 3)
 
# Perform OCR using pytesseract
text = pytesseract.image_to_string(gray)
 
# Show the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("🖼️ Input Image")
plt.axis("off")
plt.show()
 
# Print the extracted text
print("📝 Extracted Text:\n")
print(text)