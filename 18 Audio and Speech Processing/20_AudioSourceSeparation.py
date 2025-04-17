"""
Project 699: Audio Source Separation
Description:
Audio source separation refers to the task of separating individual sources (e.g., voices, instruments, or noise) from a mixed audio signal. This is important in applications like music production, speech enhancement, and audio forensics. In this project, we will implement an audio source separation system that separates a mixed audio signal (e.g., a song with vocals and accompaniment) into individual components (e.g., the vocal track and the instrumental track). We will use deep learning models like U-Net, Spleeter, or Wave-U-Net for this task.

For this project, we will use Spleeter, a pre-trained model developed by Deezer, that performs audio source separation. It can separate audio into two sources (vocals and accompaniment) or more (e.g., vocals, drums, bass, and other instruments).

Here is how you can use Spleeter for source separation:

Install Spleeter: pip install spleeter
Use Spleeter to separate an audio file into two sources: vocals and accompaniment.

Expected Output:
The audio will be separated into two components:

vocals.wav: The isolated vocal track.

accompaniment.wav: The instrumental (accompaniment) track.

Explanation of Spleeter:
Spleeter is a deep learning model trained on a large dataset of music, and it can separate the audio into two or more sources. It has models for 2 stems (vocals + accompaniment) and 4 stems (vocals, drums, bass, and other).

The model uses a neural network architecture (like U-Net) that processes the audio spectrogram and performs source separation in the time-frequency domain.
"""

from spleeter.separator import Separator
import os
 
# 1. Initialize Spleeter Separator with the pre-trained model for 2 stems (vocals and accompaniment)
separator = Separator('spleeter:2stems')  # '2stems' separates the vocals and accompaniment
 
# 2. Separate the audio into vocals and accompaniment
def separate_audio(input_file, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # Separate the audio
    separator.separate(input_file)  # The result will be saved in the 'output' folder by default
 
    print(f"Audio separated and saved in: {output_dir}")
 
# 3. Example usage
input_file = 'path_to_your_audio_file.mp3'  # Replace with the path to your audio file
separate_audio(input_file)