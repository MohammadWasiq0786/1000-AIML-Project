"""
Project 690: Voice Conversion System
Description:
A Voice Conversion (VC) system transforms the voice of one speaker (source) into the voice of another speaker (target). The goal is to modify the source speaker's features (e.g., pitch, tone, and timbre) to match those of the target speaker, while preserving the content of the speech. In this project, we will implement a basic voice conversion system using feature extraction (e.g., MFCC), speech synthesis (e.g., WaveNet or Vocoder), and simple transformations such as pitch shifting or spectral manipulation to convert a speaker's voice.

For simplicity, we will use pitch shifting to perform basic voice conversion. For more advanced systems, neural networks like CycleGANs or Autoencoders are used for high-quality conversion, but here we will focus on a basic technique using librosa.

In this Voice Conversion System, we use pitch shifting as a simple technique to change the pitch of the source speaker's voice to match that of a target speaker. More sophisticated techniques could involve deep learning models or advanced signal processing methods.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load the source audio signal
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Apply pitch shifting for voice conversion
def pitch_shift(audio, sr, n_steps):
    """
    Perform pitch shifting on the audio signal.
    :param audio: Input audio signal
    :param sr: Sample rate of the audio
    :param n_steps: Number of semitones to shift (positive for higher pitch, negative for lower pitch)
    :return: Pitch-shifted audio signal
    """
    shifted_audio = librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)
    return shifted_audio
 
# 3. Plot original and converted audio for comparison
def plot_audio_comparison(original_audio, converted_audio, sr):
    time = np.arange(len(original_audio)) / sr
 
    plt.figure(figsize=(10, 6))
 
    plt.subplot(2, 1, 1)
    plt.plot(time, original_audio, label="Original Audio")
    plt.title("Original Audio Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(time, converted_audio, label="Converted Audio", color='orange')
    plt.title("Converted Audio (Pitch Shifted)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
 
    plt.tight_layout()
    plt.show()
 
# 4. Example usage
source_file = "path_to_source_audio.wav"  # Replace with the path to the source audio file (speaker's voice)
target_pitch_shift = 2  # Shift the pitch by 2 semitones (for example, to increase the pitch)
 
# Load the source audio
audio, sr = load_audio(source_file)
 
# Perform voice conversion by pitch shifting
converted_audio = pitch_shift(audio, sr, n_steps=target_pitch_shift)
 
# Plot the comparison of original and converted audio
plot_audio_comparison(audio, converted_audio, sr)
 
# Optionally, save the converted audio
librosa.output.write_wav("converted_audio.wav", converted_audio, sr)