"""
Project 224. Video summarization
Description:
Video summarization aims to extract key highlights or representative frames from a long video, creating a short and informative version. It's useful in surveillance review, meeting recaps, content previews, and sports highlights. In this project, we'll implement a simple method using frame sampling and scene change detection based on histogram differences.

About:
What It Does:
This project performs a simple yet effective video summarization by detecting scene changes through histogram comparison. It works well for highlight generation, trailer creation, and quick video previews. You can enhance it using deep learning-based methods like VideoBERT, Shot Boundary Detection, or LSTM models for temporal summarization.
"""
# Install required package:
# pip install opencv-python matplotlib
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# Load the input video
video_path = 'long_video.mp4'  # Replace with your video
cap = cv2.VideoCapture(video_path)
 
# Parameters for summarization
scene_threshold = 0.5  # Threshold for scene change (histogram difference)
summary_frames = []
prev_hist = None
 
frame_count = 0
 
while True:
    success, frame = cap.read()
    if not success:
        break
 
    frame_count += 1
 
    # Convert to HSV and compute histogram for scene change detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
 
    if prev_hist is not None:
        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        if diff > scene_threshold:
            # Scene changed, save the frame
            summary_frames.append(frame.copy())
    else:
        # Always add the first frame
        summary_frames.append(frame.copy())
 
    prev_hist = hist
 
# Release the video capture
cap.release()
 
# Display the summary frames
plt.figure(figsize=(15, 5))
for i, frame in enumerate(summary_frames[:5]):  # Display first 5 highlights
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i + 1)
    plt.imshow(frame_rgb)
    plt.title(f"Scene {i+1}")
    plt.axis('off')
 
plt.suptitle("Video Summarization - Key Frames", fontsize=14)
plt.tight_layout()
plt.show()
 
# Optionally save summary frames to disk
for idx, frame in enumerate(summary_frames):
    cv2.imwrite(f"summary_frame_{idx+1}.jpg", frame)
 
print(f"\n🎞️ Extracted {len(summary_frames)} key frames as video summary.")