"""
Project 119. Optical character recognition
Description:
Optical Character Recognition (OCR) is the process of converting images of typed or handwritten text into machine-encoded text. It enables automatic data extraction from scanned documents, receipts, and license plates. In this project, we use Tesseract OCR with Python and OpenCV to detect and extract text from images.

About:
📸 What This Project Demonstrates:
Converts images into readable text using Tesseract OCR

Performs preprocessing (grayscale, threshold, denoise) to improve results

Can be used on scanned documents, forms, IDs, or printed receipts
"""

# Install if not already: pip install pytesseract opencv-python pillow
# Also install Tesseract OCR binary from: https://github.com/tesseract-ocr/tesseract
 
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
 
# Path to image
image_path = "sample_text.png"  # Replace with your image path
 
# Load and preprocess the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Optional: Thresholding and denoising to improve OCR accuracy
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)
 
# OCR using pytesseract
text = pytesseract.image_to_string(gray)
 
# Show original and extracted text
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("📄 Original Image")
plt.show()
 
print("\n📝 Extracted Text:\n")
print(text)