"""
Project 798: Edge NLP Applications
Description
Edge NLP applications bring natural language processing directly to low-power or offline devices — enabling features like voice commands, keyword detection, or sentiment analysis without cloud dependency. In this project, we simulate a lightweight intent recognition system that runs locally using a small LSTM model and preprocessed text commands.

✅ Edge-friendly tips:

Convert this to TensorFlow Lite for real-time inference.

Deploy on voice-enabled devices, offline assistants, or low-power embedded systems.

Expand with more training data and intents, or compress further using distillation or quantization-aware training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# Simulated user voice/text commands
sentences = [
    "turn on the lights", "switch off the fan", "play music", "stop music",
    "increase volume", "decrease volume", "what’s the weather", "set an alarm",
    "open the door", "close the window"
]
 
labels = [
    "lights_on", "fan_off", "music_play", "music_stop",
    "volume_up", "volume_down", "weather_query", "set_alarm",
    "open_door", "close_window"
]
 
# Encode intent labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
 
# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=5)
 
# Build intent classifier model (LSTM-based)
model = models.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=5),
    layers.LSTM(32),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')  # intent classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded, y, epochs=100, verbose=0)
 
# Test with new sample
test_cmd = ["turn off the lights"]
test_seq = tokenizer.texts_to_sequences(test_cmd)
test_pad = pad_sequences(test_seq, maxlen=5)
pred = np.argmax(model.predict(test_pad), axis=1)
intent = encoder.inverse_transform(pred)[0]
 
print(f"✅ Detected Intent: '{intent}' for command: \"{test_cmd[0]}\"")