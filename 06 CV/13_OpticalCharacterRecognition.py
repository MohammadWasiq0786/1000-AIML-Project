"""
Project 213. Optical character recognition
Description:
Optical Character Recognition (OCR) is the process of detecting and extracting text from images or scanned documents. It's widely used for digitizing printed documents, reading signs in images, or automating data entry. In this project, we'll use Tesseract OCR via the pytesseract Python wrapper to read and extract text from an image.

About:
What It Does:
This OCR project allows you to turn any image with text into editable digital content — whether it's scanned books, invoices, handwritten notes, or signs in photos. It's heavily used in document digitization, invoice processing, data extraction, and AI-based search systems.
"""

# Install dependencies first:
# pip install pytesseract pillow opencv-python
# Also install Tesseract OCR engine on your system:
# For Windows: https://github.com/tesseract-ocr/tesseract
# For Ubuntu: sudo apt install tesseract-ocr
 
import pytesseract
from PIL import Image
import cv2
import matplotlib.pyplot as plt
 
# Path to the image containing text
image_path = "text_image.png"  # Replace with your image
image = cv2.imread(image_path)
 
# Convert image to RGB (PIL expects RGB format)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Optional: Enhance text readability (grayscale and thresholding)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
 
# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(thresh)
 
# Display the image and the extracted text
plt.imshow(rgb_image)
plt.title("Input Image with Text")
plt.axis('off')
plt.show()
 
print("\n📝 Extracted Text:")
print(extracted_text)