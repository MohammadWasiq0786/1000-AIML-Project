"""
Project 689: Text-to-Speech Synthesis
Description:
Text-to-Speech (TTS) synthesis is the process of converting written text into spoken words. It has applications in various fields, including virtual assistants, assistive technologies, and audiobooks. In this project, we will implement a basic TTS system using Google Text-to-Speech (gTTS), which converts input text into speech. The system will generate audio files from given text inputs, which can then be played back to the user.

In this Text-to-Speech synthesis project, we use Google Text-to-Speech (gTTS) to convert a given text string into an audio file and save it as an MP3 file. The speech is generated using the English language, but you can change the language parameter for different languages (e.g., 'es' for Spanish, 'de' for German).
"""

from gtts import gTTS
import os
 
# 1. Define the text-to-speech synthesis function
def text_to_speech(text, language='en', output_file='output.mp3'):
    """
    Convert text to speech and save the output as an audio file.
    :param text: Text to convert to speech
    :param language: Language for speech synthesis (default is English)
    :param output_file: The name of the output audio file
    """
    # Initialize gTTS object with text and language
    tts = gTTS(text=text, lang=language, slow=False)  # slow=False makes the speech faster
 
    # Save the generated speech to an audio file
    tts.save(output_file)
    print(f"Speech saved as {output_file}")
 
    # Play the generated speech (optional)
    os.system(f"start {output_file}")  # This works on Windows. On macOS use 'afplay', on Linux 'mpg321' or 'aplay'
 
# 2. Example usage
text = "Hello, welcome to the Text-to-Speech project. I hope you enjoy learning!"
text_to_speech(text)  # Convert the text to speech and save as 'output.mp3'