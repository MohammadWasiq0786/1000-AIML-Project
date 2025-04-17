"""
Project 681: Speech Recognition System
Description:
A speech recognition system converts spoken language into text. It can be used for a wide range of applications such as voice assistants, transcription services, and accessibility tools. In this project, we will implement a basic speech recognition system using a pre-trained model like Google's Speech-to-Text API or a library such as SpeechRecognition to transcribe audio into text.
"""

import speech_recognition as sr
 
# 1. Initialize the recognizer
recognizer = sr.Recognizer()
 
# 2. Capture the audio from the microphone
with sr.Microphone() as source:
    print("Please say something...")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    audio = recognizer.listen(source)  # Capture the audio
 
# 3. Recognize the speech using Google Speech Recognition API
try:
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio.")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition service.")