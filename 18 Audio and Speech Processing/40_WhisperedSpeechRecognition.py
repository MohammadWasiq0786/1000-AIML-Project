"""
Project 719: Whispered Speech Recognition
Description:
Whispered speech recognition involves transcribing speech that is produced without vocal cord vibration, which makes it quieter and harder to recognize compared to normal speech. Whispered speech often lacks the normal harmonics and has different acoustic features, making traditional speech recognition models less effective. In this project, we will build a whispered speech recognition system using transfer learning and data augmentation to make the model more robust to the acoustic characteristics of whispered speech.

We will fine-tune a pre-trained model like Wav2Vec 2.0 using a dataset containing whispered speech (e.g., Whispered Speech Corpus or any available dataset with whispered speech). The model will be trained to better handle the acoustic properties of whispered speech.

Required Libraries:
pip install transformers datasets torchaudio librosa

Explanation:
Pre-trained Wav2Vec 2.0 Model: We load a pre-trained Wav2Vec 2.0 model, which has been trained on general speech data. This model is fine-tuned on a whispered speech dataset to improve its performance on whispered speech.

Dataset: In this example, we use the CommonVoice dataset, but you can replace it with a whispered speech dataset (e.g., Whispered Speech Corpus) for more accurate results.

Fine-Tuning: The pre-trained model is fine-tuned on the whispered speech dataset to adapt it to the unique characteristics of whispered speech (e.g., lower volume, lack of vocal cord vibration).

Speech Recognition: After fine-tuning, the model is used to transcribe whispered speech from new audio files.

For better results, you would need a large whispered speech dataset. Additionally, data augmentation (e.g., noise addition, pitch shifting) can help improve model robustness.
"""



import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import torchaudio
 
# 1. Load the pre-trained Wav2Vec 2.0 model and processor
def load_wav2vec_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model
 
# 2. Load and prepare the whispered speech dataset (replace with real whispered speech dataset)
def load_whispered_speech_dataset():
    # For demonstration purposes, using a common voice dataset
    # Replace with an actual whispered speech dataset if available
    dataset = load_dataset("common_voice", "en")  # Replace with a whispered speech dataset
    return dataset
 
# 3. Fine-tune the pre-trained model on the whispered speech dataset
def fine_tune_model(model, processor, train_dataset, test_dataset):
    def preprocess_function(examples):
        audio = examples["audio"]
        waveform, _ = torchaudio.load(audio["path"])
        return processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
 
    # Preprocess the dataset
    train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio"])
    test_dataset = test_dataset.map(preprocess_function, remove_columns=["audio"])
 
    # Fine-tune the model (example loop, real training would use a proper trainer)
    training_args = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
 
    for epoch in range(3):  # Training for 3 epochs (for simplicity)
        for batch in train_dataset:
            inputs = batch['input_values']
            labels = batch['labels']
            outputs = model(input_values=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            training_args.step()
            training_args.zero_grad()
 
    return model
 
# 4. Perform speech-to-text for a new whispered speech audio file
def transcribe_audio(model, processor, audio_file):
    waveform, _ = torchaudio.load(audio_file)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
 
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription
 
# 5. Example usage
# Load pre-trained Wav2Vec 2.0 model and processor
processor, model = load_wav2vec_model()
 
# Load the whispered speech dataset (replace with actual whispered speech dataset)
whispered_speech_dataset = load_whispered_speech_dataset()  # Replace with actual dataset
train_dataset = whispered_speech_dataset["train"]
test_dataset = whispered_speech_dataset["test"]
 
# Fine-tune the model on the whispered speech dataset
fine_tuned_model = fine_tune_model(model, processor, train_dataset, test_dataset)
 
# Test the fine-tuned model with a new whispered speech audio file
audio_file = "path_to_whispered_speech_audio.wav"  # Replace with your audio file path
transcription = transcribe_audio(fine_tuned_model, processor, audio_file)
print(f"Transcription of Whispered Speech: {transcription}")