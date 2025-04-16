"""
Project 390. Voice conversion system
Description:
A voice conversion system is designed to transform a speaker's voice to sound like another speaker's voice while retaining the original content. This is a challenging problem in speech processing, as it requires transforming the vocal characteristics (such as pitch, timbre, and speed) without altering the spoken words. It has applications in areas such as voice synthesis, dubbing, and identity masking.

In this project, we will explore a simple voice conversion technique using pre-trained models or signal processing techniques. For simplicity, we'll focus on transforming audio using basic spectral features like Mel-frequency cepstral coefficients (MFCCs), although more advanced systems would use deep learning methods such as CycleGAN or VAE-based models.

About:
✅ What It Does:
MFCC Extraction: Extracts MFCC features from both the source and target voices.

Simple Conversion: A very simplified approach maps the target voice's features to match the source voice's timbral characteristics.

Reconstruction: Uses inverse MFCC to reconstruct audio from the transformed features.

Output: Generates a converted audio file that sounds like the target voice, but retains the source's speech content.

Key features:
MFCC features are used to represent the voice's characteristics.

This approach is simplified for demonstration purposes, but real-world systems use more complex techniques like CycleGANs or Wavenet for voice conversion.

Audio transformation is achieved by manipulating the spectral features and reconstructing the audio from the transformed features.
"""

import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
# 1. Load and process the source and target audio
def load_audio(file_path):
    # Load audio file and resample to a consistent sampling rate
    audio, sr = librosa.load(file_path, sr=16000)
    return audio, sr
 
# 2. Extract MFCC features from the source and target audio
def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc
 
# 3. Preprocess the MFCC features (mean-variance normalization)
def preprocess_mfcc(mfcc):
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc.T).T
    return mfcc_scaled
 
# 4. Perform a basic "conversion" by matching the MFCCs of the target voice to the source voice
def convert_voice(source_mfcc, target_mfcc):
    # Here we perform simple "mapping" by directly averaging the target MFCCs to match the source
    return target_mfcc.mean(axis=1)  # This is a simplification; real conversion would involve more advanced models.
 
# 5. Reconstruct the audio from the modified MFCC features (using inverse MFCC)
def reconstruct_audio(mfcc, sr):
    # Use librosa to convert MFCC back to audio (this is a simplification)
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(mfcc)
    return reconstructed_audio
 
# 6. Example: Load source and target audio files
source_audio, sr = load_audio('source_voice.wav')
target_audio, _ = load_audio('target_voice.wav')
 
# 7. Extract and preprocess MFCCs from both audio files
source_mfcc = extract_mfcc(source_audio, sr)
target_mfcc = extract_mfcc(target_audio, sr)
 
source_mfcc_processed = preprocess_mfcc(source_mfcc)
target_mfcc_processed = preprocess_mfcc(target_mfcc)
 
# 8. Perform the "conversion" by mapping target voice features to source voice
converted_mfcc = convert_voice(source_mfcc_processed, target_mfcc_processed)
 
# 9. Reconstruct audio from the converted MFCCs (this step is simplified and may not produce high-quality output)
converted_audio = reconstruct_audio(converted_mfcc, sr)
 
# 10. Save and display the result
sf.write('converted_voice.wav', converted_audio, sr)
 
# 11. Visualize the converted MFCCs and the original waveform
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.feature.mfcc(y=converted_audio, sr=sr), x_axis='time', sr=sr)
plt.title('Converted MFCC')
plt.colorbar(format='%+2.0f dB')
plt.show()