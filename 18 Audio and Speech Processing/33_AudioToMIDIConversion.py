"""
Project 712: Audio-to-MIDI Conversion
Description:
Audio-to-MIDI conversion is the process of converting an audio signal (e.g., a music recording) into MIDI format. MIDI (Musical Instrument Digital Interface) is a standard protocol used to represent musical data, such as notes, velocities, and timings. The task is challenging because it involves converting continuous audio waveforms into discrete musical events, such as notes and chords, while maintaining timing and pitch information. In this project, we will implement an audio-to-MIDI conversion system that extracts pitch and timing information from an audio signal and generates a corresponding MIDI file.

We will use Librosa to extract pitch information from the audio signal and mido to create a MIDI file. The process will involve detecting pitch and timing of the audio and converting them into MIDI events.

Install Required Libraries:
pip install librosa mido numpy

Explanation:
Audio Feature Extraction: We extract the pitch of the audio using Librosa's pitch tracking function and the onset strength for detecting note onsets.

MIDI Conversion: We convert the extracted pitch values to MIDI note numbers (MIDI uses note numbers, with 69 representing A4) and generate MIDI events (note on and note off).

MIDI File Generation: Using the mido library, we create a MIDI file that represents the audio as musical notes with proper timings.

This is a simplified version of audio-to-MIDI conversion. For better accuracy, more advanced techniques such as deep learning models trained on large music datasets can be used for higher-quality pitch and onset detection.
"""

import librosa
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Extract pitch and timing (onset) information from the audio
def extract_pitch_and_onsets(audio, sr):
    # Detect pitch using librosa's pitch tracking algorithm
    onset_env = librosa.onset.onset_strength(audio, sr=sr)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    
    # Use librosa's pitch detection
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    
    # Extract the most prominent pitch at each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        pitch = pitches[:, t]
        index = np.argmax(pitch)  # Find the most prominent pitch
        pitch_values.append(pitch[index])
    
    return pitch_values, times
 
# 3. Convert pitch and timing information to MIDI events
def audio_to_midi(pitches, times, output_file="output_midi.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Loop through each pitch and its corresponding time
    for pitch, time in zip(pitches, times):
        # Convert pitch to MIDI note (note 69 is A4)
        midi_note = int(librosa.hz_to_midi(pitch))
        
        # Add MIDI note on event (velocity of 64)
        track.append(Message('note_on', note=midi_note, velocity=64, time=int(time * 1000)))  # time in ms
        # Add note off event (duration of 500ms for simplicity)
        track.append(Message('note_off', note=midi_note, velocity=64, time=int((time + 0.5) * 1000)))  # 500ms duration
    
    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")
 
# 4. Example usage
audio_file = 'path_to_audio_file.wav'  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Extract pitch and onset information
pitches, times = extract_pitch_and_onsets(audio, sr)
 
# Convert audio to MIDI
audio_to_midi(pitches, times, output_file="converted_audio.mid")