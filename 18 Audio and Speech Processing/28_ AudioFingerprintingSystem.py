"""
🧠 Project 707: Audio Fingerprinting System
Description:
Audio fingerprinting is a technique used to create a unique identifier (or "fingerprint") for an audio signal, which can be used for applications like audio matching, music recognition, and content identification. The goal is to generate a compact, unique representation of an audio signal that can be compared to others in a large database to identify the audio. In this project, we will implement a basic audio fingerprinting system using spectrograms and hashing to create audio fingerprints and perform audio matching.

We'll create audio fingerprints by generating spectrograms for the audio signals and then apply hashing to convert the spectrogram into a compact, unique fingerprint. We will then match fingerprints to identify audio.

Explanation:
In this audio fingerprinting system:

Spectrogram Generation: We create a mel spectrogram from the audio signal using Librosa, which provides a time-frequency representation of the audio.

Fingerprint Generation: We flatten the spectrogram into a 1D array and then generate a hash (SHA-256) from it. This hash serves as the fingerprint for the audio.

Matching: The generated fingerprint is compared with a list of known fingerprints to determine whether a match exists. In real-world scenarios, the fingerprints are stored in a large database, and the system performs fast matching.

For a production system, locality-sensitive hashing (LSH) or other techniques can be used to improve the matching efficiency.
"""

import librosa
import numpy as np
import hashlib
import matplotlib.pyplot as plt
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Create a spectrogram for the audio
def create_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    # Generate a mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to decibels for better representation
    spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return spectrogram_db
 
# 3. Generate an audio fingerprint from the spectrogram
def generate_fingerprint(spectrogram):
    # Flatten the spectrogram to a 1D array
    flattened_spectrogram = spectrogram.flatten()
    
    # Convert the array into a string to generate a unique hash
    spectrogram_str = ''.join([str(val) for val in flattened_spectrogram])
    
    # Generate a hash from the spectrogram string
    fingerprint = hashlib.sha256(spectrogram_str.encode('utf-8')).hexdigest()
    
    return fingerprint
 
# 4. Match the audio fingerprint with an existing database (for simplicity, we compare with a single known fingerprint)
def match_fingerprint(fingerprint, known_fingerprints):
    if fingerprint in known_fingerprints:
        return "Audio Matched!"
    else:
        return "No Match Found."
 
# 5. Visualize the spectrogram (optional)
def plot_spectrogram(spectrogram):
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='viridis', origin='lower', aspect='auto')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.show()
 
# 6. Example usage
audio_file = 'path_to_audio_file.wav'  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Create the spectrogram for the audio
spectrogram = create_spectrogram(audio, sr)
 
# Generate the audio fingerprint
fingerprint = generate_fingerprint(spectrogram)
 
# Visualize the spectrogram
plot_spectrogram(spectrogram)
 
# Compare the generated fingerprint with known fingerprints
known_fingerprints = ['known_fingerprint_1', 'known_fingerprint_2', fingerprint]  # Replace with real known fingerprints
match_result = match_fingerprint(fingerprint, known_fingerprints)
print(match_result)