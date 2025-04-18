"""
Project 956. Cross-modal Biometric Verification

Cross-modal biometric verification is the process of verifying an individual’s identity using data from multiple modalities, such as face recognition, voice recognition, and even fingerprints or iris scans. In this project, we simulate cross-modal biometric verification by combining facial features and voice features to verify if they belong to the same person.

In this implementation, we will use facial features from images and voice features from audio to compare the person’s identity, simulating a cross-modal verification process.

Step 1: Facial Feature Extraction
We will use OpenCV to extract basic facial features from images.

Step 2: Voice Feature Extraction
We will use DeepSpeech to extract speech features (such as text length and pitch) from audio.

Step 3: Cross-modal Verification
We will compare both facial and voice features to verify whether they match or not.

What This Does:
Facial Feature Extraction: Extracts facial features from an image using OpenCV's Haar cascades. In real-world systems, you would use deep learning-based models (like FaceNet) for more accurate and robust face embeddings.

Voice Feature Extraction: Extracts speech features (text length and pitch) from the audio using DeepSpeech. More advanced systems would use MFCCs or Wav2Vec embeddings for voice feature extraction.

Cross-modal Verification: Compares face features and voice features using cosine similarity to determine if they belong to the same person.
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
 
# Step 1: Facial Feature Extraction using OpenCV (Haar Cascades)
def extract_face_features(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    if len(faces) == 0:
        return None  # No face detected
 
    # Extracting features from the first detected face
    (x, y, w, h) = faces[0]
    face_region = image[y:y + h, x:x + w]
    face_features = np.mean(face_region)  # Simplified feature (mean intensity)
    return face_features
 
# Step 2: Extracting voice features using DeepSpeech
def extract_voice_features(audio_file):
    spoken_text = speech_to_text(audio_file)
    # Here, we use text length and pitch as simplified voice features
    text_length = len(spoken_text.split())
    fs, audio = wavfile.read(audio_file)
    pitch = np.mean(audio)  # Simplified feature: average amplitude
    return np.array([text_length, pitch])
 
# Step 3: Compare face and voice features
def compare_face_and_voice(image_path, audio_file):
    face_features = extract_face_features(image_path)
    voice_features = extract_voice_features(audio_file)
    
    if face_features is None:
        return "Face not detected in the image."
 
    # Compute similarity (cosine similarity for feature comparison)
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