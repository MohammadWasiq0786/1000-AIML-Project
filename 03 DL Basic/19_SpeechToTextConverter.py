"""
Project 99. Speech-to-text converter
Description:
A speech-to-text converter transforms spoken audio into written text. In this project, we use Python’s speech_recognition library to capture voice input from a microphone or audio file and transcribe it into text using Google Speech Recognition API.

About:
🎤 What This Project Demonstrates:
Captures speech using microphone or file input

Converts speech into text using Google Speech API

Easy and fast way to build voice-enabled apps
"""

# Install if not already: pip install SpeechRecognition pyaudio
 
import speech_recognition as sr
 
# Initialize recognizer
recognizer = sr.Recognizer()
 
# Use microphone as input
with sr.Microphone() as source:
    print("🎤 Say something! I'm listening...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
 
# Recognize speech using Google Web Speech API
try:
    text = recognizer.recognize_google(audio)
    print("📝 You said:", text)
except sr.UnknownValueError:
    print("❌ Could not understand audio")
except sr.RequestError as e:
    print(f"❌ Could not request results; {e}")
    
# 🔈 Alternate: Transcribe an Audio File
# Transcribe from an audio file (e.g., WAV format)
file_path = "audio_sample.wav"
 
with sr.AudioFile(file_path) as source:
    audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print("📝 Transcription:", text)
    except Exception as e:
        print("❌ Error:", e)