"""
Project 576: Dense Video Captioning
Description:
Dense video captioning involves generating captions for multiple events or actions within a video, rather than generating a single caption for the entire video. This task requires understanding the temporal and spatial dynamics of a video. In this project, we will use a pre-trained model to generate captions for events occurring at different points in a video.
"""

from transformers import VideoGPT2Tokenizer, VideoGPT2ForConditionalGeneration
import torch
from PIL import Image
import cv2
import numpy as np
 
# 1. Load pre-trained VideoGPT2 model and tokenizer
model_name = "huggingface/video-gpt2"
model = VideoGPT2ForConditionalGeneration.from_pretrained(model_name)
tokenizer = VideoGPT2Tokenizer.from_pretrained(model_name)
 
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
 
# 4. Preprocess the frames and convert to text representation
input_frames = [Image.fromarray(frame) for frame in frame_list]
inputs = tokenizer(input_frames, return_tensors="pt", padding=True)
 
# 5. Generate dense captions for the video
outputs = model.generate(input_ids=None, decoder_start_token_id=model.config.pad_token_id, 
                         **inputs, max_length=50, num_beams=5, early_stopping=True)
 
# 6. Decode and display the generated captions
captions = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Video Captions: {captions}")