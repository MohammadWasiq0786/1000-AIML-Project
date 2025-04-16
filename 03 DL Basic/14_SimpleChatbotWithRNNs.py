"""
Project 94. Simple chatbot with RNNs
Description:
A chatbot using RNNs can generate text-based responses by learning patterns in human conversations. In this project, we build a simple seq2seq chatbot using an RNN encoder-decoder model (with LSTM layers) trained on pairs of questions and answers.

About:
💬 What This Project Demonstrates:
Builds a simple RNN-based seq2seq chatbot

Trains on a small set of question-answer pairs

Uses teacher forcing and inference models for response generation
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# Sample conversational dataset (Q-A pairs)
data = [
    ("hi", "hello"),
    ("how are you", "i'm good"),
    ("what is your name", "i am a chatbot"),
    ("bye", "goodbye"),
    ("what do you do", "i talk to people"),
    ("thanks", "you're welcome"),
    ("hello", "hi there"),
    ("how old are you", "i am timeless")
]
 
# Split into input and output
questions, answers = zip(*data)
answers = ['<sos> ' + ans + ' <eos>' for ans in answers]
 
# Tokenize input and output
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
 
input_seq = tokenizer.texts_to_sequences(questions)
target_seq = tokenizer.texts_to_sequences(answers)
 
# Padding
max_len_input = max(len(seq) for seq in input_seq)
max_len_target = max(len(seq) for seq in target_seq)
 
encoder_input_data = pad_sequences(input_seq, maxlen=max_len_input, padding='post')
decoder_input_data = pad_sequences([seq[:-1] for seq in target_seq], maxlen=max_len_target - 1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in target_seq], maxlen=max_len_target - 1, padding='post')
 
vocab_size = len(tokenizer.word_index) + 1
 
# One-hot encode decoder target
decoder_target_onehot = tf.keras.utils.to_categorical(decoder_target_data, num_classes=vocab_size)
 
# Define encoder
encoder_inputs = Input(shape=(max_len_input,))
enc_emb = tf.keras.layers.Embedding(vocab_size, 64)(encoder_inputs)
_, state_h, state_c = LSTM(64, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]
 
# Define decoder
decoder_inputs = Input(shape=(max_len_target - 1,))
dec_emb = tf.keras.layers.Embedding(vocab_size, 64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
 
# Define and compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_onehot, epochs=500, verbose=0)
 
# Inference models for prediction
encoder_model = Model(encoder_inputs, encoder_states)
 
decoder_state_input_h = Input(shape=(64,))
decoder_state_input_c = Input(shape=(64,))
dec_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = tf.keras.layers.Embedding(vocab_size, 64)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + dec_states_inputs, [decoder_outputs2] + decoder_states2)
 
# Reverse token mapping
reverse_word_index = {i: word for word, i in tokenizer.word_index.items()}
 
# Chatbot response generator
def decode_sequence(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len_input, padding='post')
    states_value = encoder_model.predict(input_seq)
 
    target_seq = np.array([[tokenizer.word_index['<sos>']]])
    stop_condition = False
    decoded_sentence = ''
 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
 
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')
 
        if sampled_word == '<eos>' or len(decoded_sentence.split()) > 10:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
 
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
 
    return decoded_sentence.strip()
 
# Test the chatbot
print("User: what is your name")
print("Bot:", decode_sequence("what is your name"))
 
print("User: how are you")
print("Bot:", decode_sequence("how are you"))
 
print("User: bye")
print("Bot:", decode_sequence("bye"))