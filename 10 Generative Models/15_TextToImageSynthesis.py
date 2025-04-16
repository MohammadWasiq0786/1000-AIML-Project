"""
Project 375. Text-to-image synthesis
Description:
Text-to-Image Synthesis is a task in which the model generates an image based on a textual description. For example, if the input is "a red apple on a table," the model will generate an image that corresponds to this description. This task typically involves generative adversarial networks (GANs) or variational autoencoders (VAEs) combined with a text encoder, such as a recurrent neural network (RNN) or transformer, to process the input text.

In this project, we will implement a basic Text-to-Image Synthesis model using GANs.

About:
✅ What It Does:
Uses BERT to convert text descriptions into embeddings, which are then fed into the Generator and Discriminator models for training

The Generator creates images conditioned on random noise and text embeddings, while the Discriminator classifies them as real or fake

Trains on the CIFAR-10 dataset with paired captions to generate 32x32 RGB images based on the textual descriptions

The text-to-image synthesis process allows generating images from textual input, e.g., "a red apple on a table"
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
 
# 1. Define the Text-to-Image GAN model
class TextToImageGenerator(nn.Module):
    def __init__(self, z_dim=100, text_embedding_dim=768, img_channels=3):
        super(TextToImageGenerator, self).__init__()
        self.fc1 = nn.Linear(z_dim + text_embedding_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, img_channels * 64 * 64)  # Generate 64x64 image
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z, text_embedding):
        # Concatenate random noise with text embeddings
        x = torch.cat((z, text_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x.view(-1, 3, 64, 64)  # Reshape to image dimensions
 
# 2. Define the Discriminator model
class TextToImageDiscriminator(nn.Module):
    def __init__(self, text_embedding_dim=768):
        super(TextToImageDiscriminator, self).__init__()
        self.fc1 = nn.Linear(3 * 64 * 64 + text_embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Output single value: real or fake
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, text_embedding):
        x = x.view(-1, 3 * 64 * 64)  # Flatten the image
        input = torch.cat((x, text_embedding), dim=1)  # Concatenate image and text
        x = self.leaky_relu(self.fc1(input))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return self.sigmoid(self.fc4(x))
 
# 3. Load pre-trained BERT model for text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel.from_pretrained("bert-base-uncased")
 
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Use the mean of last layer hidden states
 
# 4. Loss function and optimizer
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 5. Training loop for Text-to-Image GAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = TextToImageGenerator().to(device)
discriminator = TextToImageDiscriminator().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, captions) in enumerate(train_loader):  # Assumes captions are paired with images
        real_images = real_images.to(device)
        text_embeddings = text_to_embedding(captions).to(device)
 
        # Create labels for real and fake data
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
 
        # Train the Discriminator
        optimizer_d.zero_grad()
 
        # Train on real images
        real_outputs = discriminator(real_images, text_embeddings)
        d_loss_real = criterion(real_outputs, real_labels)
 
        # Train on fake images
        z = torch.randn(real_images.size(0), 100).to(device)  # Random noise
        fake_images = generator(z, text_embeddings)
        fake_outputs = discriminator(fake_images.detach(), text_embeddings)
        d_loss_fake = criterion(fake_outputs, fake_labels)
 
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
 
        # Train the Generator
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images, text_embeddings)
        g_loss = criterion(fake_outputs, real_labels)  # We want fake images to be classified as real
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
 
    # Generate and display sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(8, 100).to(device)  # Random noise
            captions = ["a red apple on a table"] * 8  # Example captions
            text_embeddings = text_to_embedding(captions).to(device)
            fake_images = generator(z, text_embeddings).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()