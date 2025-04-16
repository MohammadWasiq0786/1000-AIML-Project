"""
Project 391. Speech synthesis implementation
Description:
Speech synthesis (or text-to-speech) is the process of converting text into spoken words using a machine learning model. It's a crucial task in applications like virtual assistants, screen readers, and voice-based interaction systems. In this project, we'll implement a basic speech synthesis system using a pre-trained text-to-speech model (e.g., Tacotron or WaveGlow), which can generate human-like speech from text.

About:
✅ What It Does:
Text-to-Speech (TTS): Converts a given text into speech using a pre-trained Tacotron2 model.

Audio Output: The generated speech is saved as a WAV file.

Play Audio: Optionally plays the generated speech on a supported system.

Key features:
Pre-trained models like Tacotron2 or FastSpeech are used to generate high-quality speech.

The system allows you to convert any text into natural-sounding speech.

The audio is saved as a WAV file that can be played back or used in applications.
"""

# Install TTS library for text-to-speech synthesis
# pip install TTS
from TTS.api import TTS
import soundfile as sf
 
# 1. Load a pre-trained TTS model
# Using the TTS library, we load a pre-trained model for speech synthesis.
# You can specify the model name for the desired language and voice style.
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name=model_name)
 
# 2. Text-to-Speech Conversion
text = "Hello, welcome to the text to speech synthesis demonstration!"
speech = tts.tts(text)
 
# 3. Save the generated speech as a WAV file
output_file = "output_speech.wav"
sf.write(output_file, speech, 22050)  # Save the audio file with 22.05kHz sampling rate
 
# 4. Optionally, play the sound to hear the result
import os
os.system(f"aplay {output_file}")  # This works on Linux systems; on Windows, use any audio player