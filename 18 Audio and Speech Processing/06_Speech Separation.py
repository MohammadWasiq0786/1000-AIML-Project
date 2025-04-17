"""
Project 686: Speech Separation (Cocktail Party Problem)
Description:
The cocktail party problem refers to the challenge of separating multiple overlapping speech signals from a mixture, much like trying to hear a single conversation in a noisy room with many people talking. In this project, we will implement a speech separation system that uses a Deep Learning-based approach (e.g., Convolutional Neural Networks or U-Net) to separate mixed audio signals into individual speech sources. We will simulate this using a mixture of two speech signals and apply a separation algorithm.

To implement speech separation using a deep learning approach, such as U-Net, you'll need a dataset of mixed speech signals (e.g., LibriMix) and pre-trained models. Here's a high-level implementation using PyTorch and Wave-U-Net.

Since actual deep learning-based speech separation is quite complex and typically requires a large dataset, I'll show a simplified approach to separate two mixed signals using scipy for basic signal separation. For deep learning methods, a library like Spleeter (developed by Deezer) can be used for real-world applications.

This example shows how to simulate a cocktail party problem by mixing two speech signals and applying a basic spectral subtraction technique to attempt separation. However, for real-world use, more advanced techniques like Wave-U-Net, Spleeter, or Deep Clustering should be used for high-quality speech separation.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
 
# 1. Load two speech signals to simulate a cocktail party scenario
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Simulate a cocktail party by mixing two speech signals
def mix_speech_signals(signal1, signal2, snr_db=10):
    """
    Mix two speech signals at a specified signal-to-noise ratio (SNR).
    :param signal1: First speech signal
    :param signal2: Second speech signal
    :param snr_db: Signal-to-noise ratio in decibels
    :return: Mixed signal
    """
    # Calculate the power of both signals
    power1 = np.sum(signal1**2)
    power2 = np.sum(signal2**2)
 
    # Calculate scaling factor based on desired SNR
    snr_linear = 10 ** (snr_db / 10)
    scale_factor = np.sqrt((power1 + power2) / (snr_linear * power2))
 
    # Mix signals
    mixed_signal = signal1 + scale_factor * signal2
    return mixed_signal
 
# 3. Simple speech separation using basic spectral subtraction
def spectral_subtraction_separation(mixed_signal, original_signal1, original_signal2):
    """
    Perform a basic separation using spectral subtraction.
    :param mixed_signal: The mixed signal
    :param original_signal1: Original first signal (used for estimating noise)
    :param original_signal2: Original second signal (used for estimating noise)
    :return: Separated signals
    """
    # Compute Short-Time Fourier Transform (STFT) of the mixed signal
    f, t, Zxx = stft(mixed_signal, nperseg=1024)
 
    # Estimate the magnitude and phase of the signals
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
 
    # Estimate noise magnitude using original signals
    f, t, Zxx1 = stft(original_signal1, nperseg=1024)
    f, t, Zxx2 = stft(original_signal2, nperseg=1024)
    mag1 = np.abs(Zxx1)
    mag2 = np.abs(Zxx2)
 
    # Spectral subtraction (very simplified version)
    mag_separated1 = mag - mag1
    mag_separated2 = mag - mag2
    mag_separated1 = np.maximum(mag_separated1, 0)  # Avoid negative values
    mag_separated2 = np.maximum(mag_separated2, 0)  # Avoid negative values
 
    # Reconstruct signals
    _, separated_signal1 = istft(mag_separated1 * np.exp(1j * phase), nperseg=1024)
    _, separated_signal2 = istft(mag_separated2 * np.exp(1j * phase), nperseg=1024)
 
    return separated_signal1, separated_signal2
 
# 4. Example usage
signal1_file = "path_to_speech1.wav"  # Replace with path to first speech file
signal2_file = "path_to_speech2.wav"  # Replace with path to second speech file
 
# Load the speech signals
signal1, sr = load_audio(signal1_file)
signal2, _ = load_audio(signal2_file)
 
# Mix the speech signals with a desired SNR
mixed_signal = mix_speech_signals(signal1, signal2, snr_db=10)
 
# Separate the mixed signal into individual speech signals
separated_signal1, separated_signal2 = spectral_subtraction_separation(mixed_signal, signal1, signal2)
 
# Plot the original, mixed, and separated signals
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(signal1, label="Original Speech 1")
plt.title("Original Speech 1")
plt.subplot(3, 1, 2)
plt.plot(mixed_signal, label="Mixed Signal")
plt.title("Mixed Signal")
plt.subplot(3, 1, 3)
plt.plot(separated_signal1, label="Separated Speech 1")
plt.title("Separated Speech 1")
plt.tight_layout()
plt.show()