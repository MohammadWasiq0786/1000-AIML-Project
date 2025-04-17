"""
Project 702: Audio Inpainting System
Description:
Audio inpainting is the process of filling in missing parts of an audio signal, similar to how image inpainting works for missing pixels. This can be useful in scenarios where parts of an audio recording are corrupted or missing, such as in audio restoration or noise removal. In this project, we will implement an audio inpainting system that uses a simple neural network or interpolation methods to restore missing parts of the audio signal.

In this simple implementation, we will use linear interpolation to fill in missing audio segments. For more advanced methods, deep learning models like WaveNet or autoencoders can be used for high-quality inpainting. Here, we will demonstrate filling missing values with interpolation.

Explanation:
In this audio inpainting project, we simulate missing data by randomly setting portions of the audio to NaN (Not a Number). Then, we use linear interpolation to fill in the missing parts of the audio. The interp1d function from scipy is used to perform the interpolation. This is a basic approach for audio restoration; for better results, deep learning models such as WaveNet or autoencoders can be applied to learn the best way to restore missing audio parts.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Simulate missing data (for inpainting) by setting parts of the audio to NaN
def simulate_missing_data(audio, missing_percentage=0.2):
    # Create a mask to simulate missing audio data
    n_samples = len(audio)
    missing_samples = int(n_samples * missing_percentage)
    missing_indices = np.random.choice(n_samples, missing_samples, replace=False)
    
    # Set the selected indices to NaN to simulate missing data
    audio_with_missing = audio.copy()
    audio_with_missing[missing_indices] = np.nan
    
    return audio_with_missing, missing_indices
 
# 3. Inpaint the missing audio data using linear interpolation
def inpaint_audio(audio_with_missing):
    # Find the indices where the data is missing (NaN values)
    not_nan_indices = ~np.isnan(audio_with_missing)
    missing_indices = np.isnan(audio_with_missing)
    
    # Interpolate the missing data using linear interpolation
    f = interp1d(np.where(not_nan_indices)[0], audio_with_missing[not_nan_indices], kind='linear', fill_value="extrapolate")
    inpainted_audio = f(np.arange(len(audio_with_missing)))
    
    return inpainted_audio
 
# 4. Visualize the original, missing, and inpainted audio
def plot_audio_comparison(original_audio, audio_with_missing, inpainted_audio, sr):
    time = np.arange(len(original_audio)) / sr
 
    plt.figure(figsize=(10, 6))
 
    # Plot the original audio
    plt.subplot(3, 1, 1)
    plt.plot(time, original_audio, label="Original Audio")
    plt.title("Original Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
 
    # Plot the audio with missing data
    plt.subplot(3, 1, 2)
    plt.plot(time, audio_with_missing, label="Audio with Missing Data", color='orange')
    plt.title("Audio with Missing Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
 
    # Plot the inpainted audio
    plt.subplot(3, 1, 3)
    plt.plot(time, inpainted_audio, label="Inpainted Audio", color='green')
    plt.title("Inpainted Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
 
    plt.tight_layout()
    plt.show()
 
# 5. Example usage
audio_file = "path_to_audio.wav"  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Simulate missing data
audio_with_missing, missing_indices = simulate_missing_data(audio)
 
# Inpaint the missing audio using linear interpolation
inpainted_audio = inpaint_audio(audio_with_missing)
 
# Visualize the comparison
plot_audio_comparison(audio, audio_with_missing, inpainted_audio, sr)