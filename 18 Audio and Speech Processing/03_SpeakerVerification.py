"""
Project 683: Speaker Verification
Description:
Speaker verification is the process of verifying the identity of a speaker based on their voice. It is a type of biometric authentication that checks whether the speaker is who they claim to be. In this project, we will implement a speaker verification system that compares the voice of a person to a stored voice model (template) and decides whether the person is authenticated. We will use MFCC features for voice feature extraction and a distance-based classifier like cosine similarity for speaker verification.
"""

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfcc, axis=1)  # Use the mean of the MFCC features for verification
 
# 2. Store the voice model of a person (template)
def create_speaker_model(file_path):
    """
    Generate a speaker model by extracting the MFCC features from an audio file.
    :param file_path: The path to the audio file of the speaker.
    :return: The MFCC features representing the speaker's voice.
    """
    return extract_mfcc(file_path)
 
# 3. Verify the speaker by comparing the input voice to the stored model using cosine similarity
def verify_speaker(stored_model, input_audio_file, threshold=0.8):
    """
    Verify if the input audio matches the stored model based on cosine similarity.
    :param stored_model: MFCC features representing the stored speaker's model.
    :param input_audio_file: Path to the input audio file for verification.
    :param threshold: Cosine similarity threshold for accepting the match.
    :return: True if the speaker is verified, False otherwise.
    """
    input_model = extract_mfcc(input_audio_file)  # Extract MFCC features from the input audio
    similarity = cosine_similarity([stored_model], [input_model])  # Calculate cosine similarity
    print(f"Cosine Similarity: {similarity[0][0]}")
 
    if similarity[0][0] > threshold:
        print("Speaker verified!")
        return True
    else:
        print("Speaker not verified!")
        return False
 
# 4. Example usage
# 4.1 Create a model for the speaker by providing an audio sample
speaker_model_file = "path_to_speaker_audio.wav"  # Replace with the path to the speaker's voice file
stored_model = create_speaker_model(speaker_model_file)
 
# 4.2 Verify the speaker using a new audio sample
verification_audio_file = "path_to_input_audio.wav"  # Replace with the path to the input audio for verification
verify_speaker(stored_model, verification_audio_file)