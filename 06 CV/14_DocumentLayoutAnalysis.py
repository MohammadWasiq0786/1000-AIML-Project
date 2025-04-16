"""
Project 214. Document layout analysis
Description:
Document Layout Analysis involves detecting and categorizing structural elements (like titles, paragraphs, tables, images) within a document image. It's an essential step for document understanding, digitization, and automated data extraction. In this project, we'll use the LayoutParser library, which combines deep learning models to identify various layout components in documents.

About:
What It Does:
This project identifies the structure and layout of complex documents (like invoices, resumes, academic papers), enabling automated processing such as table extraction, section splitting, OCR by region, and more. It powers intelligent document understanding systems in fintech, healthcare, education, and enterprise automation.
"""
# Install required packages:
# pip install layoutparser[layoutmodels,tesseract] opencv-python matplotlib
 
import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
 
# Load the document image
image_path = "invoice_sample.jpg"  # Replace with your document image
image = cv2.imread(image_path)
 
# Convert to RGB (OpenCV loads in BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Load a pre-trained model for document layout analysis (PubLayNet model)
# This model can detect: text, title, list, table, figure
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],  # Confidence threshold
)
 
# Perform layout detection
layout = model.detect(image_rgb)
 
# Draw the layout results on the image
image_with_boxes = lp.draw_box(image_rgb, layout, box_width=3, show_element_type=True)
 
# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(image_with_boxes)
plt.title("Document Layout Analysis")
plt.axis('off')
plt.show()
 
# Print detected layout elements with coordinates and type
print("\n🗂️ Detected Layout Elements:")
for block in layout:
    print(f"{block.type}: {block.coordinates}")