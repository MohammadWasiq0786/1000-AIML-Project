"""
Project 98. Image captioning model
Description:
An image captioning model generates natural language descriptions for images. It combines a CNN (to extract image features) with an RNN/LSTM (to generate text). In this project, we implement a basic image captioning pipeline using a pre-trained InceptionV3 for image encoding and an LSTM decoder to generate captions.

About:
📷 What This Project Demonstrates:
Extracts image features using CNN (InceptionV3)

Generates natural language captions using RNN/LSTM

Combines vision and language models in a unified pipeline
"""

# Install if not already: pip install tensorflow pillow matplotlib
 
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add
import matplotlib.pyplot as plt
 
# Dummy tokenizer and captions (normally from dataset like Flickr8k)
captions = {
    "example.jpg": ["startseq a dog running through a field endseq"]
}
word_to_index = {"startseq": 1, "a": 2, "dog": 3, "running": 4, "through": 5, "field": 6, "endseq": 7}
index_to_word = {i: w for w, i in word_to_index.items()}
vocab_size = len(word_to_index) + 1
max_len = 10
 
# Load pre-trained CNN (InceptionV3) for image feature extraction
base_model = InceptionV3(weights='imagenet')
model_cnn = Model(base_model.input, base_model.layers[-2].output)  # Use penultimate layer
 
def extract_image_features(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model_cnn.predict(x)
 
# Define the RNN-based caption decoder model
def define_caption_model():
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)
 
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
 
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
 
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
 
model = define_caption_model()
# Normally you'd train the model here — we assume it's pretrained for demo
 
# Caption generator function (mocked prediction for simplicity)
def generate_caption(image_feature):
    in_text = 'startseq'
    for _ in range(max_len):
        sequence = [word_to_index.get(w, 0) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_len)
        # Mock prediction: just add next word
        next_index = len(sequence[0])  # Dummy step
        next_word = index_to_word.get(next_index, 'endseq')
        in_text += ' ' + next_word
        if next_word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()
 
# Extract features and generate caption
img_path = tf.keras.utils.get_file('example.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
features = extract_image_features(img_path)
caption = generate_caption(features)
 
# Show result
img = load_img(img_path)
plt.imshow(img)
plt.axis('off')
plt.title("Generated Caption: " + caption)
plt.show()