"""
Project 581: Human-Object Interaction Detection
Description:
Human-object interaction detection focuses on identifying the interactions between humans and objects within an image or video. This involves detecting both the human and the object, as well as the relationship between them (e.g., "person holding a cup"). In this project, we will use a pre-trained model to detect human-object interactions in images or videos.
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
 
# 3. Load an image for human-object interaction detection
image_path = "path_to_image.jpg"  # Replace with an actual image path
image = cv2.imread(image_path)
 
# 4. Perform object detection on the image
outputs = predictor(image)
 
# 5. Visualize the results and the detected interactions
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
result_image = v.get_image()[:, :, ::-1]
 
# 6. Display the result
cv2.imshow("Human-Object Interaction Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()