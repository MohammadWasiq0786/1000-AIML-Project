"""
Project 577: Video Question Answering
Description:
Video question answering (VQA) involves answering questions related to a video by understanding both the visual and temporal information in the video. In this project, we will use a pre-trained vision-and-language model to answer questions related to the content of a video, leveraging both the visual features and the context provided by the question.
"""

from transformers import VideoQuestionAnsweringProcessor, VideoQuestionAnsweringModel
import torch
from PIL import Image
import cv2
import numpy as np
 
# 1. Load pre-trained Video Q&A model and processor
model_name = "facebook/maskformer-swin-large"
model = VideoQuestionAnsweringModel.from_pretrained(model_name)
processor = VideoQuestionAnsweringProcessor.from_pretrained(model_name)
 
# 2. Load a video (use a video file path or URL)
video_path = "path_to_video.mp4"  # Replace with an actual video path
cap = cv2.VideoCapture(video_path)
 
# 3. Process frames from the video
frame_list = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)
cap.release()
 
# 4. Convert the frames to a format compatible with the processor
input_frames = [Image.fromarray(frame) for frame in frame_list]
 
# 5. Define a question related to the video
question = "What action is happening in the video?"
 
# 6. Preprocess the frames and question for the model
inputs = processor(text=question, images=input_frames, return_tensors="pt", padding=True)
 
# 7. Perform Video Question Answering
outputs = model(**inputs)
 
# 8. Get the answer and display it
answer = outputs["logits"]
predicted_answer = torch.argmax(answer)
print(f"Predicted Answer: {predicted_answer.item()}")