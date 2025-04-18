from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
 
class CLIPImageEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
    def get_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.squeeze().tolist()