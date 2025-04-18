"""
Project 894. Voice Authentication System

A voice authentication system verifies users based on the unique characteristics of their voice (pitch, tone, rhythm). In this project, we simulate speaker verification by extracting MFCC (Mel Frequency Cepstral Coefficients) features and comparing them using cosine similarity.

Why MFCC?
MFCCs capture the timbral texture of a voice — it's like a fingerprint for speech.

Cosine similarity is used here for speaker matching.

🔒 Advanced systems include:

Noise filtering & silence trimming

Dynamic Time Warping (DTW)

Deep speaker embedding models (e.g., ECAPA-TDNN, x-vectors)
"""

import librosa
import numpy as np
from scipy.spatial.distance import cosine
 
# Function to extract voice features (MFCC)
def extract_voice_features(filepath):
    audio, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Average across time
    return mfcc_mean
 
# Load and extract features from enrolled and test samples
enrolled_features = extract_voice_features("enrolled_voice.wav")
input_features = extract_voice_features("input_voice.wav")
 
# Compute cosine similarity
similarity = 1 - cosine(enrolled_features, input_features)
threshold = 0.85  # Empirical similarity threshold
 
print(f"Voice Similarity Score: {similarity:.2f}")
if similarity >= threshold:
    print("✅ Voice Verified: Access Granted.")
else:
    print("❌ Voice Mismatch: Access Denied.")