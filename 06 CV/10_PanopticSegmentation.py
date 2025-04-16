"""
Project 210. Panoptic segmentation implementation
Description:
Panoptic segmentation is a combination of semantic and instance segmentation. It assigns every pixel in an image a class label and distinguishes between different instances of things (like cars, people) while also labeling stuff (like road, sky, grass). We'll use Detectron2, a powerful library by Facebook AI, which supports panoptic segmentation with pre-trained models.

About:
What It Does:
This implementation gives you both semantic context (stuff) and individual objects (things) in one shot. It's ideal for advanced scene understanding, autonomous driving, and mixed reality applications where everything in the scene needs to be identified and separated meaningfully.
"""

# Install Detectron2 before running:
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.2/index.html
 
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
 
# Load image using OpenCV (Detectron2 uses BGR format)
image_path = "street_scene.jpg"  # Replace with your image
image = cv2.imread(image_path)
 
# Set up Detectron2 config for panoptic segmentation with a ResNet-FPN backbone
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5  # Filter weak instances
 
# Create the predictor using the loaded config
predictor = DefaultPredictor(cfg)
 
# Run inference on the image
outputs = predictor(image)  # Result contains 'panoptic_seg' and 'segments_info'
 
# Extract segmentation map and segments metadata
panoptic_seg, segments_info = outputs["panoptic_seg"]
 
# Visualize panoptic segmentation
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
 
# Convert back to displayable format
result_image = out.get_image()
 
# Display the result using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.title("Panoptic Segmentation Output")
plt.axis('off')
plt.show()