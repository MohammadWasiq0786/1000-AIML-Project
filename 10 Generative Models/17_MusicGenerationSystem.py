"""
Project 377. Music generation system
Description:
A music generation system uses generative models to create melodies or full music tracks based on learned patterns in musical data. Models such as Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Transformer networks can generate music by predicting the next note in a sequence of musical events.

In this project, we will implement a simple LSTM-based music generation system to generate melodies or short music pieces.

About:
✅ What It Does:
Defines an LSTM model for generating music based on a sequence of notes

Uses music21 to convert MIDI files into sequences of notes/chords and prepares them for training

Trains the model to predict the next note in a sequence based on previous notes

Generates music by sampling from the trained model and converting the output into musical notation using music21

Saves the generated music as a MIDI file
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import music21  # Music21 for working with music notation
from music21 import converter, instrument, note, chord, stream
import random
 
# 1. Define the LSTM model for music generation
class MusicGenerationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicGenerationModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.fc(lstm_out)
        return output
 
# 2. Prepare and process music data
def prepare_music_data(midi_file_path):
    # Convert the MIDI file to music21 stream
    midi_stream = converter.parse(midi_file_path)
    
    # Extract note and chord data
    notes = []
    for element in midi_stream.flat.notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    
    # Create a unique set of notes/chords and a mapping to integers
    unique_notes = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for note, number in note_to_int.items()}
    
    # Convert notes to integers
    sequence = [note_to_int[note] for note in notes]
    
    return sequence, note_to_int, int_to_note
 
# 3. Train the model
def train_music_model(sequence, input_size=1, hidden_size=256, num_layers=2, output_size=128, num_epochs=50):
    model = MusicGenerationModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare the sequence data
    seq_len = 100  # Number of previous notes to consider for prediction
    input_data = []
    target_data = []
    
    for i in range(len(sequence) - seq_len):
        input_data.append(sequence[i:i+seq_len])
        target_data.append(sequence[i+seq_len])
    
    input_data = torch.tensor(input_data, dtype=torch.float32)
    target_data = torch.tensor(target_data, dtype=torch.long)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(input_data)
        loss = criterion(output.view(-1, output_size), target_data)
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model
 
# 4. Generate music from trained model
def generate_music(model, start_sequence, int_to_note, length=500):
    model.eval()
    
    # Start with the input sequence
    input_seq = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    generated_notes = []
    
    for _ in range(length):
        with torch.no_grad():
            prediction = model(input_seq)
            predicted_note = torch.argmax(prediction).item()
            generated_notes.append(predicted_note)
            input_seq = torch.cat((input_seq[:, 1:], prediction[:, -1:].unsqueeze(1)), dim=1)
    
    # Convert generated notes to actual musical notation
    output_notes = []
    for note_int in generated_notes:
        note_str = int_to_note[note_int]
        if '.' in note_str or note_str.isdigit():  # It's a chord
            chord_notes = note_str.split('.')
            chord_notes = [note.Note(int(n)) for n in chord_notes]
            output_notes.append(chord.Chord(chord_notes))
        else:  # It's a note
            output_notes.append(note.Note(note_str))
    
    return output_notes
 
# 5. Convert generated notes to a music21 stream and save as MIDI
def save_music_to_midi(output_notes, output_path="generated_music.mid"):
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_path)
 
# 6. Example usage: Prepare music data, train the model, and generate music
sequence, note_to_int, int_to_note = prepare_music_data('path_to_your_midi_file.mid')
model = train_music_model(sequence)
 
# Generate music starting from the first 100 notes in the sequence
generated_notes = generate_music(model, sequence[:100], int_to_note)
 
# Save generated music as a MIDI file
save_music_to_midi(generated_notes)