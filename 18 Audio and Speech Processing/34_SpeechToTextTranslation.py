"""
Project 713: Speech-to-Text Translation
Description:
Speech-to-text translation is the process of converting spoken language into written text. This is a fundamental task in speech recognition systems and is widely used in applications like virtual assistants, dictation software, and voice-enabled devices. In this project, we will implement a speech-to-text system that takes an audio input (speech) and converts it into corresponding text using a pre-trained model like Google Speech-to-Text API or DeepSpeech.

In this implementation, we will use the Google Speech-to-Text API to perform speech recognition. The API converts audio files containing speech into text with high accuracy. You'll need a Google Cloud account and set up the Google Speech-to-Text API.

Steps:
Install the required libraries:

pip install SpeechRecognition pydub
Google Cloud setup:

Go to Google Cloud Console, enable the Speech-to-Text API, and create a service account with a key.

Download the JSON key and set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to the downloaded file.

Explanation:
Audio Conversion: We use pydub to convert the audio file (e.g., MP3) to a WAV file with mono channel, 16kHz sample rate, and 16-bit depth. These settings are required by the Google Speech-to-Text API.

Speech Recognition: The SpeechRecognition library is used to send the audio data to the Google Speech API for transcription. The recognize_google method sends the audio data to Google’s API and retrieves the corresponding text.

Error Handling: We handle potential errors like unrecognizable speech or API request failures.

This system uses Google's Speech-to-Text API, but you can replace it with other speech recognition systems such as DeepSpeech or AssemblyAI for better customization or offline recognition.
"""

import speech_recognition as sr
from pydub import AudioSegment
 
# 1. Load and convert the audio file to the required format (WAV)
def load_and_convert_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)  # Mono, 16 kHz, 16-bit
    audio.export("converted_audio.wav", format="wav")
    return "converted_audio.wav"
 
# 2. Perform speech-to-text using Google Speech API
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    
    # Load the audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    
    try:
        # Use Google Web Speech API to recognize speech
        print("Recognizing...")
        text = recognizer.recognize_google(audio_data)
        print(f"Recognized Text: {text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
 
# 3. Example usage
audio_file = 'path_to_audio_file.mp3'  # Replace with your audio file path
 
# Convert the audio file to the required format
converted_audio = load_and_convert_audio(audio_file)
 
# Perform speech-to-text
speech_to_text(converted_audio)