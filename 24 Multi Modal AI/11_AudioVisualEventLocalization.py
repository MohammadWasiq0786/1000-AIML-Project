"""
Project 931. Audio-Visual Event Localization

Audio-visual event localization involves identifying and locating events in video that are triggered by both audio and visual signals. This can be used in applications like video surveillance, activity detection, and multi-modal content analysis. In this project, we simulate an audio-visual event localization system using audio recognition and object detection.

What This Does:
Audio Event Detection: Uses librosa to detect audio events (e.g., speech onset times) based on the onset strength.

Visual Event Localization: Uses OpenCV Haar Cascades to detect faces in video frames.

Event Synchronization: Combines the audio and visual events by matching the frame numbers with detected audio events.
"""

import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
 
# Step 1: Audio Event Detection using librosa
def detect_audio_event(audio_file):
    y, sr = librosa.load(audio_file)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Simple event detection: find peaks in the onset envelope (audio events)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    print(f"Detected audio events at times: {peaks}")
    return peaks
 
# Step 2: Visual Event Localization using OpenCV (Haar Cascades)
def detect_faces_in_video(video_file):
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    face_locations = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_locations.append((frame_count, x, y, w, h))  # Store frame count and face position
 
        frame_count += 1
 
    cap.release()
    print(f"Detected faces in frames: {face_locations}")
    return face_locations
 
# Example usage:
audio_file = "example_audio.wav"  # Replace with a valid audio file
video_file = "example_video.mp4"  # Replace with a valid video file
 
# Detect audio events (e.g., speech or music)
audio_events = detect_audio_event(audio_file)
 
# Detect visual events (e.g., faces) in the video
face_locations = detect_faces_in_video(video_file)
 
# Step 3: Combine audio and visual event data (e.g., synchronization)
# Example: If a face is detected in a frame when an audio event occurs, we localize it
for event_time in audio_events:
    event_frame = int(event_time * 30)  # Assume 30 fps for video
    print(f"Audio event at {event_time}s corresponds to frame {event_frame}.")
 
    # Check if there are faces in this frame
    faces_in_frame = [loc for loc in face_locations if loc[0] == event_frame]
    if faces_in_frame:
        print(f"Faces detected in frame {event_frame}: {faces_in_frame}")