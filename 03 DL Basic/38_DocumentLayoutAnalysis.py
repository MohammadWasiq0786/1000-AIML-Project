"""
Project 118. Document layout analysis
Description:
Document Layout Analysis involves detecting and classifying various components of a document such as text blocks, tables, images, headers, and footers. This is a crucial step in digitizing scanned documents, invoices, forms, and books. In this project, we use LayoutParser, a powerful library that leverages deep learning models to perform layout detection on documents.

About:
📄 What This Project Demonstrates:
Detects layout components like text, titles, tables, figures

Uses deep learning-based object detection models (Detectron2 via LayoutParser)

Enables structured extraction from unstructured documents
"""

# Install if not already: pip install layoutparser[layoutmodels] opencv-python matplotlib
 
import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
 
# Load document image
image_path = "sample_doc.png"  # Replace with your own scanned document
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Load a pre-trained model for layout detection (PubLayNet model)
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],  # Set threshold
    use_gpu=False  # Set to True if CUDA is available
)
 
# Perform layout analysis
layout = model.detect(image)
 
# Draw layout blocks on image
image_with_blocks = lp.draw_box(image_rgb, layout, box_width=3, show_element_type=True)
 
# Display result
plt.figure(figsize=(12, 10))
plt.imshow(image_with_blocks)
plt.title("Document Layout Detection")
plt.axis("off")
plt.tight_layout()
plt.show()
 
# Optional: extract text blocks for OCR or processing
text_blocks = [b for b in layout if b.type == 'Text']
for i, block in enumerate(text_blocks):
    x_1, y_1, x_2, y_2 = map(int, block.coordinates)
    cropped = image[y_1:y_2, x_1:x_2]
    # Here, you could apply OCR or save the region
    cv2.imwrite(f"text_block_{i}.png", cropped)