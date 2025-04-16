"""
Project 217. License plate recognition
Description:
License Plate Recognition (LPR) involves detecting and extracting text from vehicle license plates. It's widely used in traffic monitoring, toll systems, parking access, and law enforcement. This project combines object detection (to locate the plate) and OCR (to read the characters). We'll use OpenCV for plate detection and Tesseract OCR to extract the plate number.

About:
What It Does:
This project detects the license plate area in an image and reads the plate number using OCR. It's a mini version of an automatic number plate recognition (ANPR) system, useful for smart parking, highway tolls, security checkpoints, and urban traffic systems.
"""
# Install dependencies:
# pip install pytesseract opencv-python pillow
 
import cv2
import pytesseract
from matplotlib import pyplot as plt
 
# Load the vehicle image
image_path = "car.jpg"  # Replace with your vehicle image
image = cv2.imread(image_path)
 
# Convert to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Use OpenCV's built-in method to enhance edges and contours
# Helps isolate the license plate region
blurred = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduces noise while preserving edges
edged = cv2.Canny(blurred, 30, 200)              # Perform edge detection
 
# Find contours in the edged image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# Sort contours by area and keep top 10
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
 
license_plate = None
 
# Loop through contours to find a rectangle that might be a license plate
for c in contours:
    approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
    if len(approx) == 4:  # Look for a rectangular contour
        license_plate = approx
        break
 
# Mask everything except the license plate
mask = cv2.drawContours(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), [license_plate], -1, 255, -1)
x, y, w, h = cv2.boundingRect(license_plate)
plate_img = gray[y:y+h, x:x+w]
 
# Use pytesseract to read text from the cropped plate region
plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')  # Assume a single word line
 
# Draw rectangle and label on original image
image_display = image.copy()
cv2.drawContours(image_display, [license_plate], -1, (0, 255, 0), 3)
cv2.putText(image_display, f'Plate: {plate_text.strip()}', (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
 
# Convert BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
 
# Show results
plt.figure(figsize=(12, 6))
plt.imshow(image_rgb)
plt.title(f"License Plate Detected: {plate_text.strip()}")
plt.axis('off')
plt.show()
 
# Print final plate text
print("\n🚗 Detected License Plate Number:")
print(plate_text.strip())