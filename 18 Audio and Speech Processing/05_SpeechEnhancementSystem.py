"""
Project 685: Speech Enhancement System
Description:
A speech enhancement system improves the quality and clarity of speech signals, especially in noisy environments. This is important for applications such as voice recognition, telecommunication, and assistive devices. In this project, we will implement a simple speech enhancement system that removes background noise from a speech signal using spectral subtraction or Wiener filtering. The goal is to improve the quality of the speech signal by reducing noise while preserving speech details.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
 
# 1. Load the noisy speech signal
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Apply spectral subtraction to enhance speech
def spectral_subtraction(noisy_signal, noise_estimate):
    """
    Perform spectral subtraction for speech enhancement.
    :param noisy_signal: Noisy speech signal
    :param noise_estimate: Estimated noise signal
    :return: Enhanced speech signal
    """
    # Perform STFT (Short-Time Fourier Transform) to get frequency-domain representation
    noisy_stft = librosa.stft(noisy_signal)
    noise_stft = librosa.stft(noise_estimate)
 
    # Subtract the noise spectrum from the noisy signal's spectrum
    enhanced_stft = noisy_stft - noise_stft
    enhanced_stft = np.maximum(enhanced_stft, 0)  # Ensure no negative values in the spectrum
 
    # Perform inverse STFT to get the enhanced time-domain signal
    enhanced_signal = librosa.istft(enhanced_stft)
    return enhanced_signal
 
# 3. Visualize the original and enhanced signals
def plot_signals(noisy_signal, enhanced_signal, sr):
    plt.figure(figsize=(10, 6))
 
    # Plot the noisy signal
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal)
    plt.title("Noisy Speech Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
 
    # Plot the enhanced signal
    plt.subplot(2, 1, 2)
    plt.plot(enhanced_signal)
    plt.title("Enhanced Speech Signal (After Spectral Subtraction)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
 
    plt.tight_layout()
    plt.show()
 
# 4. Example usage
noisy_file = "path_to_noisy_audio.wav"  # Replace with the path to the noisy speech file
noise_file = "path_to_noise_audio.wav"  # Replace with the path to a noise sample
 
# Load the audio signals
noisy_signal, sr = load_audio(noisy_file)
noise_signal, _ = load_audio(noise_file)
 
# Apply spectral subtraction to enhance the speech signal
enhanced_signal = spectral_subtraction(noisy_signal, noise_signal)
 
# Visualize the original and enhanced signals
plot_signals(noisy_signal, enhanced_signal, sr)
 
# Optionally, save the enhanced signal
librosa.output.write_wav("enhanced_speech.wav", enhanced_signal, sr)