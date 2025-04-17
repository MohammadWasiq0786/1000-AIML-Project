"""
Project 562: Few-shot Object Detection
Description:
Few-shot object detection involves detecting objects in images with very few labeled examples. This is a challenging problem in computer vision as most object detection models require a large amount of labeled data to perform well. In this project, we will explore techniques for few-shot learning in object detection, using models such as Meta-RCNN or Detection Transformers (DETR).
"""

import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
 
# 1. Setup configuration for Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the score threshold for predictions
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Number of object classes in COCO
 
# 2. Initialize the predictor
predictor = DefaultPredictor(cfg)
 
# 3. Load an image for object detection
image_path = "path_to_image.jpg"  # Replace with an actual image path
image = cv2.imread(image_path)
 
# 4. Perform object detection on the image
outputs = predictor(image)
 
# 5. Visualize the results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
result_image = v.get_image()[:, :, ::-1]
 
# 6. Show the result
cv2.imshow("Few-shot Object Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()