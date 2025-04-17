"""
Project 687: Voice Activity Detection
Description:
Voice Activity Detection (VAD) is the process of detecting the presence or absence of human speech in an audio signal. This is a crucial preprocessing step for speech-related applications like speech recognition, audio compression, and telecommunication. In this project, we will implement a VAD system that identifies speech segments in an audio signal and separates them from non-speech segments (e.g., silence or noise). The system will use energy-based detection or spectral features like MFCC to distinguish between speech and non-speech segments.

This VAD system uses an energy-based method to detect voice activity by calculating the energy of the signal in short frames. If the energy in a frame exceeds a predefined threshold, it is considered as speech (voice activity); otherwise, it is labeled as non-speech (silence or noise).
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
 
# 1. Load the audio signal
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Voice Activity Detection using energy-based method
def voice_activity_detection(audio, frame_size=1024, threshold=0.02):
    """
    Perform Voice Activity Detection (VAD) using an energy-based method.
    :param audio: Input audio signal
    :param frame_size: Size of the frame for energy calculation (samples)
    :param threshold: Energy threshold for detecting speech activity
    :return: A binary array (1 for speech, 0 for non-speech)
    """
    # Compute the energy of the signal in short frames
    energy = np.array([np.sum(np.abs(audio[i:i+frame_size])**2) for i in range(0, len(audio), frame_size)])
 
    # Normalize energy for consistent thresholding
    energy = energy / np.max(energy)
 
    # Detect voice activity based on energy threshold
    vad = (energy > threshold).astype(int)
    return vad, energy
 
# 3. Plot the audio signal and VAD result
def plot_vad(audio, vad, energy, sr, frame_size=1024):
    """
    Plot the audio signal with the Voice Activity Detection (VAD) results.
    :param audio: Audio signal
    :param vad: VAD binary array (1 for speech, 0 for non-speech)
    :param energy: Energy of the audio signal
    :param sr: Sample rate
    :param frame_size: Size of the frame for energy calculation (samples)
    """
    time = np.arange(len(audio)) / sr
    time_frames = np.arange(0, len(audio), frame_size) / sr
 
    # Plot audio signal
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, audio, label="Audio Signal")
    plt.title("Audio Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
 
    # Plot VAD result
    plt.subplot(2, 1, 2)
    plt.plot(time_frames, energy, label="Energy", color="orange")
    plt.step(time_frames, vad, where='post', label="VAD (Speech/Non-speech)", color="blue", linestyle='--')
    plt.title("Voice Activity Detection (VAD)")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()
 
# 4. Example usage
audio_file = "path_to_audio.wav"  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Apply VAD
vad, energy = voice_activity_detection(audio, frame_size=1024, threshold=0.02)
 
# Plot the audio signal and VAD result
plot_vad(audio, vad, energy, sr)

