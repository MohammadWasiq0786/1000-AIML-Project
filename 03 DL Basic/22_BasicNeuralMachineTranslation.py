"""
Project 102. Basic neural machine translation
Description:
Neural Machine Translation (NMT) is the task of automatically translating text from one language to another using neural networks. In this project, we implement a basic sequence-to-sequence (seq2seq) model with LSTM encoder-decoder architecture in Keras to translate short phrases from English to French.

About:
🌍 What This Project Demonstrates:
Builds a basic seq2seq NMT model from scratch

Uses LSTM encoder-decoder architecture

Translates English to French using a custom vocabulary
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# Sample bilingual sentence pairs
pairs = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    ("i am fine", "je vais bien"),
    ("see you later", "à plus tard"),
    ("what is your name", "comment tu t'appelles"),
    ("i love you", "je t'aime"),
    ("goodbye", "au revoir"),
    ("yes", "oui")
]
 
# Split into source and target
eng_sentences, fr_sentences = zip(*pairs)
fr_sentences = ['<sos> ' + text + ' <eos>' for text in fr_sentences]
 
# Tokenize English
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_sentences)
eng_seq = eng_tokenizer.texts_to_sequences(eng_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
max_eng_len = max(len(seq) for seq in eng_seq)
X = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
 
# Tokenize French
fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(fr_sentences)
fr_seq = fr_tokenizer.texts_to_sequences(fr_sentences)
fr_vocab_size = len(fr_tokenizer.word_index) + 1
max_fr_len = max(len(seq) for seq in fr_seq)
 
# Prepare decoder input and target sequences
decoder_input_seq = [seq[:-1] for seq in fr_seq]
decoder_target_seq = [seq[1:] for seq in fr_seq]
 
decoder_input = pad_sequences(decoder_input_seq, maxlen=max_fr_len - 1, padding='post')
decoder_target = pad_sequences(decoder_target_seq, maxlen=max_fr_len - 1, padding='post')
decoder_target = tf.keras.utils.to_categorical(decoder_target, num_classes=fr_vocab_size)
 
# Define the NMT model
embedding_dim = 64
lstm_units = 128
 
# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = tf.keras.layers.Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
_, state_h, state_c = LSTM(lstm_units, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]
 
# Decoder
decoder_inputs = Input(shape=(max_fr_len - 1,))
dec_emb = tf.keras.layers.Embedding(fr_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(fr_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
 
# Compile and train
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X, decoder_input], decoder_target, epochs=500, verbose=0)
 
# Inference models for translation
encoder_model = Model(encoder_inputs, encoder_states)
 
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb_inf = tf.keras.layers.Embedding(fr_vocab_size, embedding_dim)(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(dec_emb_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf)
 
# Reverse lookup dictionary
reverse_fr_index = {i: w for w, i in fr_tokenizer.word_index.items()}
 
# Translate function
def translate(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    states_value = encoder_model.predict(seq)
 
    target_seq = np.array([[fr_tokenizer.word_index['<sos>']]])
    decoded = []
 
    for _ in range(max_fr_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        next_idx = np.argmax(output_tokens[0, -1, :])
        next_word = reverse_fr_index.get(next_idx, '')
        if next_word == '<eos>' or next_word == '':
            break
        decoded.append(next_word)
        target_seq = np.array([[next_idx]])
        states_value = [h, c]
 
    return ' '.join(decoded)
 
# Example translation
print("EN: how are you")
print("FR:", translate("how are you"))
 
print("\nEN: i love you")
print("FR:", translate("i love you"))