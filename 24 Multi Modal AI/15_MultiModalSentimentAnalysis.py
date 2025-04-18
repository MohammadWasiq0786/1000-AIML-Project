"""
Project 935. Multi-modal Sentiment Analysis

Multi-modal sentiment analysis combines data from multiple modalities, such as text and audio, to assess the sentiment expressed. For example, in a conversation, the tone of voice and text content can both influence the sentiment of a speaker.

In this project, we simulate multi-modal sentiment analysis by analyzing both the textual content (from user input) and the tone of voice (from audio). We’ll use TextBlob for sentiment analysis on text and librosa to extract audio features like pitch and tempo for sentiment analysis based on voice tone.

Step 1: Sentiment Analysis from Text
We'll use TextBlob to perform sentiment analysis on the textual content.

Step 2: Sentiment Analysis from Audio
We'll use librosa to extract pitch and tempo features and perform sentiment analysis based on voice tone.

What This Does:
Text Sentiment Analysis: Uses TextBlob to analyze the sentiment polarity (positive, negative, or neutral) of the text.

Audio Sentiment Analysis: Extracts pitch and tempo features using librosa, and uses simple thresholds to determine if the tone of voice is positive, negative, or neutral.

Multi-modal Sentiment: Combines the results from both text and audio to generate an overall sentiment.
"""

import librosa
import numpy as np
from textblob import TextBlob
import torch
import librosa.display
 
# Step 1: Sentiment Analysis from Text using TextBlob
def text_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"
 
# Example text input
text_input = "I love this product, it's amazing!"
text_sentiment_result = text_sentiment(text_input)
print(f"Text Sentiment: {text_sentiment_result}")
 
# Step 2: Sentiment Analysis from Audio using librosa
def audio_sentiment(audio_file):
    y, sr = librosa.load(audio_file)
    
    # Extract pitch (fundamental frequency) and tempo features from audio
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
 
    # Analyze the features to determine sentiment (simplified logic)
    avg_pitch = np.mean(pitch[pitch > 0])  # average pitch (higher is often more positive)
    avg_tempo = tempo  # tempo (higher tempo may indicate excitement)
 
    if avg_pitch > 200 and avg_tempo > 120:
        return "Positive"
    elif avg_pitch < 100 and avg_tempo < 100:
        return "Negative"
    else:
        return "Neutral"
 
# Example audio input (replace with a valid audio file path)
audio_input = "example_audio.wav"
audio_sentiment_result = audio_sentiment(audio_input)
print(f"Audio Sentiment: {audio_sentiment_result}")
 
# Combine both text and audio sentiment results
final_sentiment = "Overall Sentiment: " + ("Positive" if text_sentiment_result == "Positive" and audio_sentiment_result == "Positive" else "Negative" if text_sentiment_result == "Negative" or audio_sentiment_result == "Negative" else "Neutral")
print(final_sentiment)