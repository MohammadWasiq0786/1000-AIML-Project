"""
Project 460. Medical speech recognition
Description:
Medical Speech Recognition enables automatic transcription of doctor-patient conversations, radiology dictations, or clinical notes using ASR (Automatic Speech Recognition) systems. In this project, we'll build a prototype using a pretrained Whisper model to convert medical audio into text.

✅ What It Does:
Loads OpenAI's Whisper model for speech-to-text.

Converts spoken medical notes into transcribed reports.

Can be extended to:

Use Whisper-large or wav2vec2-medical for higher accuracy

Add speaker diarization (e.g., doctor vs patient)

Deploy for real-time dictation tools in hospitals

For production:

Use OpenAI Whisper, wav2vec 2.0, or NVIDIA NeMo.

Train or fine-tune on MediSpeech, MTSamples, or MIMIC-IV-Note datasets.
"""

import torch
import whisper
 
# 1. Load Whisper model (base model for general transcription)
model = whisper.load_model("base")
 
# 2. Transcribe an audio file (WAV/MP3/M4A)
# Replace 'medical_note.mp3' with your clinical audio file path
result = model.transcribe("medical_note.mp3")
 
# 3. Print transcribed text
print("Transcribed Medical Speech:\n")
print(result["text"])