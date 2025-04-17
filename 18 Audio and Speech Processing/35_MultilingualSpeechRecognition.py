"""
Project 714: Multilingual Speech Recognition
Description:
Multilingual speech recognition refers to the task of converting spoken language into text in multiple languages. This is crucial in applications like voice assistants, global communication tools, and real-time translation systems. In this project, we will implement a multilingual speech recognition system that can handle speech in different languages. We'll use a pre-trained model or API that supports multiple languages, such as Google Cloud Speech-to-Text or DeepSpeech.

We'll use the Google Cloud Speech-to-Text API, which supports multiple languages, including English, Spanish, French, German, and many others. The implementation will automatically detect the language of the speech, but you can also specify the language if you know it in advance.

Steps:
Install the required libraries:

pip install google-cloud-speech pydub SpeechRecognition
Google Cloud setup:

Go to Google Cloud Console, enable the Speech-to-Text API, and create a service account with a key.

Download the JSON key and set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to the downloaded file.

Explanation:
Audio Conversion: We use pydub to convert the audio file (e.g., MP3) to WAV format with mono channel, 16kHz sample rate, and 16-bit depth, as required by the Google Cloud Speech API.

Google Cloud Speech-to-Text API: The Google Cloud Speech API is used to transcribe audio into text. It supports multiple languages, and you can specify the language for speech recognition by passing the language_code parameter (e.g., "en-US" for English, "es-ES" for Spanish).

Language Detection: In this example, you can specify the language directly. However, the Google API can auto-detect the language if needed.

Supported Languages by Google Cloud:
English (en-US)

Spanish (es-ES)

French (fr-FR)

German (de-DE)

Chinese (zh-CN, zh-TW)

Many others.

For a real-world multilingual system, you would likely need to handle language detection automatically or allow the user to choose the language.
"""

import os
import speech_recognition as sr
from google.cloud import speech
from pydub import AudioSegment
 
# 1. Set up the Google Cloud Speech client
def initialize_google_cloud_client():
    client = speech.SpeechClient()
    return client
 
# 2. Convert audio to WAV format
def load_and_convert_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)  # Mono, 16 kHz, 16-bit
    audio.export("converted_audio.wav", format="wav")
    return "converted_audio.wav"
 
# 3. Perform multilingual speech recognition using Google Cloud Speech API
def recognize_speech(file_path, language_code="en-US"):
    client = initialize_google_cloud_client()
    
    with open(file_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )
    
    # Perform the recognition request
    response = client.recognize(config=config, audio=audio)
    
    # Print the recognized text from each segment
    for result in response.results:
        print(f"Recognized Text: {result.alternatives[0].transcript}")
 
# 4. Example usage
audio_file = "path_to_audio_file.mp3"  # Replace with the path to your audio file
 
# Convert audio file to WAV format
converted_audio = load_and_convert_audio(audio_file)
 
# Recognize speech in the audio file (specify language code)
language_code = "es-ES"  # Spanish (you can change this to other languages like "fr-FR", "de-DE", etc.)
recognize_speech(converted_audio, language_code)