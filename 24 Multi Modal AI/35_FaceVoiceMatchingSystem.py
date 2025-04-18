"""
Project 955. Face-Voice Matching System

A Face-Voice Matching System is designed to match an individual's face to their voice, which can be applied in scenarios such as identity verification or security systems. The system uses both facial features (from an image) and speech features (from an audio clip) to verify if both the face and voice belong to the same person.

In this project, we simulate face-voice matching by extracting facial features from an image and voice features from an audio clip. We then compute their similarity using deep learning models or simple feature comparison techniques.

Step 1: Facial Feature Extraction
We use OpenCV for face detection and face recognition.

Step 2: Voice Feature Extraction
We use DeepSpeech to convert speech into text and extract acoustic features (like pitch and tempo) for matching.

Step 3: Matching Face and Voice
We combine face features and voice features to compute the similarity score.

What This Does:
Facial Feature Extraction: Detects a face in the image using OpenCV Haar cascades and extracts simple features (e.g., average intensity).

Voice Feature Extraction: Converts speech into text using DeepSpeech and extracts basic features like text length and pitch as proxies for voice features.

Matching Face and Voice: Compares the extracted features using cosine similarity to determine if the face and voice belong to the same person.
"""

import deepspeech
import numpy as np
import cv2
from scipy.io import wavfile
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
 
# Load DeepSpeech model for speech-to-text conversion
speech_model = deepspeech.Model("deepspeech-0.9.3-models.pbmm")
 
# Function for speech-to-text conversion
def speech_to_text(audio_file):
    fs, audio = wavfile.read(audio_file)
    audio_input = np.array(audio, dtype=np.float32)
    return speech_model.stt(audio_input)
 
# Step 1: Face recognition using OpenCV
def extract_face_features(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    if len(faces) == 0:
        return None  # No face detected
 
    # Simplified: Just extract features from the first detected face
    (x, y, w, h) = faces[0]
    face_region = image[y:y + h, x:x + w]
    face_features = np.mean(face_region)  # Simple average intensity as feature (can be improved with deep models)
    return face_features
 
# Step 2: Extract speech features
def extract_voice_features(audio_file):
    spoken_text = speech_to_text(audio_file)
    # Here, we simply use text length and pitch as a proxy for voice features (in real systems, more features like MFCC are used)
    text_length = len(spoken_text.split())
    fs, audio = wavfile.read(audio_file)
    pitch = np.mean(audio)  # Simplified feature: average amplitude (for demo purposes)
    return np.array([text_length, pitch])
 
# Step 3: Compare face and voice features
def compare_face_and_voice(image_path, audio_file):
    face_features = extract_face_features(image_path)
    voice_features = extract_voice_features(audio_file)
    
    if face_features is None:
        return "Face not detected in the image."
    
    # Compute similarity (cosine similarity for demonstration)
    similarity_score = cosine_similarity([face_features], [voice_features])
    return similarity_score[0][0]
 
# Example inputs
image_path = "person_image.jpg"  # Replace with a valid image path
audio_file = "person_audio.wav"  # Replace with a valid audio file
 
# Step 4: Perform face and voice matching
similarity_score = compare_face_and_voice(image_path, audio_file)
print(f"Face-Voice Similarity Score: {similarity_score:.2f}")
 
# Final decision based on similarity score
if similarity_score > 0.7:
    print("Face and voice match, likely same person.")
else:
    print("Face and voice do not match, different person.")