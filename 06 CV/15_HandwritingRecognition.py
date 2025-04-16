"""
Project 215. Handwriting recognition
Description:
Handwriting recognition is the task of identifying and converting handwritten text into machine-readable digital text. It’s used in digitizing notes, reading forms, and extracting handwritten data. In this project, we’ll use the IAM handwriting dataset-compatible model via transformers — specifically the TrOCR model (Transformer OCR) developed by Microsoft.

About:
What It Does:
This project allows AI to read and transcribe handwritten text, whether from scanned notes, forms, or personal writing samples. It's especially useful in note digitization, form automation, education tools, and historical document analysis.
"""
# Install dependencies first:
# pip install transformers torch torchvision pillow
 
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
 
# Load an image containing handwritten text
image_path = "handwriting_sample.jpg"  # Replace with your image
image = Image.open(image_path).convert("RGB")
 
# Load the TrOCR processor and model (base model trained on IAM handwritten data)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
 
# Preprocess the image and prepare for model input
pixel_values = processor(images=image, return_tensors="pt").pixel_values
 
# Generate output (decode the handwritten content)
with torch.no_grad():
    generated_ids = model.generate(pixel_values)
 
# Decode the predicted text tokens
recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
 
# Display the image and the recognized text
plt.imshow(image)
plt.title(f"Recognized Handwriting:\n{recognized_text}", fontsize=12)
plt.axis('off')
plt.show()
 
# Also print the text in the terminal
print("\n✍️ Recognized Text:")
print(recognized_text)