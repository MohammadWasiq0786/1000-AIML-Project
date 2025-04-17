"""
Project 688: Automatic Speech Segmentation
Description:
Speech segmentation is the process of dividing continuous speech into meaningful units such as words, phrases, or sentences. In this project, we will implement an automatic speech segmentation system that segments a speech signal into different speech regions. This system will rely on voice activity detection (VAD) to differentiate between speech and non-speech segments, then apply boundary detection techniques to identify where individual speech events begin and end.

This speech segmentation system uses voice activity detection (VAD) to detect speech and non-speech segments and then applies boundary detection to identify the start and end of speech regions. The system visualizes the audio signal, its energy, and the detected speech segments.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
 
# 1. Load the audio signal
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Voice Activity Detection (VAD) to identify speech segments
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
 
# 3. Detect speech segment boundaries from VAD
def detect_speech_boundaries(vad):
    """
    Detect boundaries between speech and non-speech regions.
    :param vad: Binary VAD array (1 for speech, 0 for non-speech)
    :return: List of speech segment start and end indices
    """
    boundaries = []
    is_speech = False
    start = None
 
    for i in range(len(vad)):
        if vad[i] == 1 and not is_speech:  # Speech starts
            start = i
            is_speech = True
        elif vad[i] == 0 and is_speech:  # Speech ends
            boundaries.append((start, i - 1))
            is_speech = False
 
    # If the last segment is speech, add it
    if is_speech:
        boundaries.append((start, len(vad) - 1))
 
    return boundaries
 
# 4. Visualize the speech segments
def plot_speech_segmentation(audio, vad, energy, sr, boundaries, frame_size=1024):
    """
    Visualize the speech segmentation, showing the audio signal, energy, and identified speech segments.
    :param audio: Audio signal
    :param vad: VAD binary array (1 for speech, 0 for non-speech)
    :param energy: Energy of the audio signal
    :param sr: Sample rate
    :param boundaries: List of detected speech segment boundaries
    :param frame_size: Size of the frame for energy calculation (samples)
    """
    time = np.arange(len(audio)) / sr
    time_frames = np.arange(0, len(audio), frame_size) / sr
 
    # Plot audio signal
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, audio, label="Audio Signal")
    plt.title("Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
 
    # Plot energy
    plt.subplot(3, 1, 2)
    plt.plot(time_frames, energy, label="Energy", color="orange")
    plt.title("Energy of the Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
 
    # Plot speech segments
    plt.subplot(3, 1, 3)
    plt.plot(time, audio, label="Audio Signal", alpha=0.5)
    for start, end in boundaries:
        plt.axvspan(start / sr, (end + 1) / sr, color='green', alpha=0.5, label="Speech Segment")
    plt.title("Speech Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
 
# 5. Example usage
audio_file = "path_to_audio.wav"  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Apply VAD to detect speech regions
vad, energy = voice_activity_detection(audio, frame_size=1024, threshold=0.02)
 
# Detect speech boundaries
boundaries = detect_speech_boundaries(vad)
 
# Plot the segmentation results
plot_speech_segmentation(audio, vad, energy, sr, boundaries)