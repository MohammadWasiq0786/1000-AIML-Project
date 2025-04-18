"""
Project 927. Multi-modal Emotion Recognition

Multi-modal emotion recognition involves analyzing both audio and visual cues (like facial expressions and voice tone) to detect emotions. This can be applied to applications like sentiment analysis, human-computer interaction, and customer service bots.

In this project, we simulate a multi-modal emotion recognition system by combining audio features (from speech) and visual features (from facial expressions).

Here’s the Python implementation using both audio (via DeepSpeech for transcription and basic sentiment analysis) and visual (via OpenCV for basic face emotion detection):

Step 1: Facial Emotion Detection (Visual)
We’ll use OpenCV and deepface (a simple face recognition library) to detect emotions from the facial expressions.

What This Does:
Visual Emotion Recognition: Uses DeepFace to detect dominant emotions from facial expressions (e.g., happy, sad, angry).

Audio Emotion Recognition: Uses DeepSpeech to transcribe the audio, followed by TextBlob to perform sentiment analysis on the transcription (positive, negative, or neutral).

Multi-modal Analysis: Combines the emotions detected from both facial expressions and speech to produce a comprehensive emotion profile.
"""

import cv2
import numpy as np
from deepface import DeepFace
from textblob import TextBlob
import deepspeech
from scipy.io import wavfile
 
# Load DeepSpeech model for audio transcription
speech_model = deepspeech.Model("deepspeech-0.9.3-models.pbmm")
 
# Load video and audio file
video_file = "example_video.mp4"
audio_file = "example_audio.wav"
 
# Function to detect emotion from facial expressions using DeepFace
def detect_face_emotion(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']
 
# Function to transcribe audio using DeepSpeech
def transcribe_audio(audio_file):
    fs, audio = wavfile.read(audio_file)
    audio_input = np.array(audio, dtype=np.float32)
    return speech_model.stt(audio_input)
 
# Step 1: Extract frames and analyze face emotions
cap = cv2.VideoCapture(video_file)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
 
while True:
    success, frame = cap.read()
    if not success:
        break
 
    # Detect emotion from the frame
    emotion = detect_face_emotion(frame)
    print(f"Frame {frame_count} - Detected Emotion: {emotion}")
 
    frame_count += 1
 
cap.release()
 
# Step 2: Transcribe audio and detect sentiment
audio_transcription = transcribe_audio(audio_file)
print(f"Audio Transcription: {audio_transcription}")
 
# Perform sentiment analysis on the transcribed audio
sentiment = TextBlob(audio_transcription).sentiment.polarity
if sentiment > 0:
    audio_emotion = "Positive"
elif sentiment < 0:
    audio_emotion = "Negative"
else:
    audio_emotion = "Neutral"
 
print(f"Audio Sentiment: {audio_emotion}")
 
# Combine both audio and visual emotion detection results
final_emotion = f"Visual Emotion: {emotion}, Audio Sentiment: {audio_emotion}"
print(f"Final Combined Emotion: {final_emotion}")