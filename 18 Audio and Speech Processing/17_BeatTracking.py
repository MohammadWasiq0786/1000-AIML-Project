"""
Project 696: Beat Tracking Implementation
Description:
Beat tracking is the task of detecting the rhythmic beats in a musical signal. It is important for music analysis, DJing software, and music synchronization. In this project, we will implement a beat tracking system that detects the beats in a music track using onset detection and tempo estimation techniques. We will use librosa to extract rhythmic features and implement a simple beat tracking algorithm.

In this Beat Tracking Implementation, we use onset detection to identify the locations of beats in a music track. The librosa library's beat_track function estimates the tempo and locates the beats in the signal. The system visualizes the audio signal and its detected beats, allowing us to assess the accuracy of the beat tracking process.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Perform onset detection to detect the beats
def beat_tracking(audio, sr):
    # Perform onset detection (detect the onset of beats in the signal)
    onset_env = librosa.onset.onset_strength(audio, sr=sr)  # Calculate onset envelope
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)  # Estimate tempo and beats
    return tempo, beats, onset_env
 
# 3. Plot the audio signal with beat locations
def plot_beats(audio, beats, onset_env, sr):
    plt.figure(figsize=(10, 6))
 
    # Plot the audio signal
    plt.subplot(2, 1, 1)
    plt.plot(audio)
    plt.title("Audio Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
 
    # Plot onset envelope
    plt.subplot(2, 1, 2)
    plt.plot(onset_env, label="Onset Envelope")
    plt.vlines(beats, 0, np.max(onset_env), color='r', label="Detected Beats")
    plt.title("Onset Envelope and Detected Beats")
    plt.xlabel("Frames")
    plt.ylabel("Amplitude")
    plt.legend()
 
    plt.tight_layout()
    plt.show()
 
# 4. Example usage
audio_file = "path_to_music_audio.wav"  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Track beats using onset detection
tempo, beats, onset_env = beat_tracking(audio, sr)
 
# Print the detected tempo and beats
print(f"Estimated Tempo: {tempo} BPM")
print(f"Detected Beats: {beats}")
 
# Plot the audio signal with the detected beats
plot_beats(audio, beats, onset_env, sr)