"""
Project 697: Music Generation System
Description:
Music generation is the process of creating new musical pieces, either from scratch or based on existing musical patterns. This is a fundamental task in AI-driven music composition, creative applications, and personalized music creation. In this project, we will implement a music generation system using Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks to generate music based on learned patterns from a dataset of musical compositions. We will use a basic melody dataset and train the model to generate melodies.

In this Music Generation System, we use an LSTM (Long Short-Term Memory) neural network to generate new music. The model is trained on MIDI files, where it learns the patterns of musical notes (chords, melodies) and generates a sequence of notes. The system uses MFCC features and chromagram for feature extraction from the raw audio and creates a MIDI file from the generated sequence.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import music21  # A library for working with music notation
import random
 
# 1. Prepare the dataset (e.g., MIDI files converted to music21 format)
def prepare_dataset(midi_files):
    notes = []
    for file in midi_files:
        midi = music21.converter.parse(file)
        for element in midi.flat.notes:
            if isinstance(element, music21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, music21.note.Rest):
                notes.append('rest')
    return notes
 
# 2. Preprocess data for LSTM input
def preprocess_data(notes):
    # Create a sorted list of unique notes
    unique_notes = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for number, note in enumerate(unique_notes)}
    
    # Convert the notes into integer sequences
    input_sequences = []
    output_notes = []
    sequence_length = 100  # Length of input sequence for the LSTM
 
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_sequences.append([note_to_int[note] for note in sequence_in])
        output_notes.append(note_to_int[sequence_out])
 
    # Reshape the data into the format required by LSTM
    X = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))
    X = X / float(len(unique_notes))  # Normalize the input
    y = tf.keras.utils.to_categorical(output_notes, num_classes=len(unique_notes))
 
    return X, y, int_to_note
 
# 3. Build the LSTM model for music generation
def build_model(X, y, sequence_length, n_notes):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(n_notes, activation='softmax'))  # Output layer for note probabilities
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
 
# 4. Train the music generation model
def train_model(model, X, y, epochs=200):
    model.fit(X, y, epochs=epochs, batch_size=64)
    return model
 
# 5. Generate new music based on the trained model
def generate_music(model, int_to_note, sequence_length, seed_notes, n_generate=500):
    # Generate a sequence of notes based on the seed notes
    generated_notes = []
    for _ in range(n_generate):
        prediction_input = np.reshape(seed_notes, (1, len(seed_notes), 1))
        prediction_input = prediction_input / float(len(int_to_note))  # Normalize the input
        predicted_probs = model.predict(prediction_input, verbose=0)
        index = np.argmax(predicted_probs)
        predicted_note = int_to_note[index]
 
        # Add the predicted note to the generated notes list
        generated_notes.append(predicted_note)
        seed_notes.append(index)  # Add the prediction to the input sequence
        seed_notes = seed_notes[1:]
 
    return generated_notes
 
# 6. Convert the generated notes back to music21 format and save as a MIDI file
def create_midi(generated_notes, output_file='generated_music.mid'):
    stream = music21.stream.Stream()
    for note in generated_notes:
        if note == 'rest':
            stream.append(music21.note.Rest())
        else:
            stream.append(music21.note.Note(note))
    stream.write('midi', fp=output_file)
 
# 7. Example usage
midi_files = ['path_to_midi_file_1.mid', 'path_to_midi_file_2.mid']  # Replace with paths to MIDI files
notes = prepare_dataset(midi_files)  # Prepare the dataset
X, y, int_to_note = preprocess_data(notes)  # Preprocess the data
 
# Build and train the model
model = build_model(X, y, sequence_length=100, n_notes=len(int_to_note))
model = train_model(model, X, y, epochs=200)
 
# Generate new music
seed_notes = [random.choice(range(len(int_to_note)))] * 100  # Starting with a random seed
generated_notes = generate_music(model, int_to_note, sequence_length=100, seed_notes=seed_notes, n_generate=500)
 
# Save the generated music to a MIDI file
create_midi(generated_notes, output_file='generated_music.mid')