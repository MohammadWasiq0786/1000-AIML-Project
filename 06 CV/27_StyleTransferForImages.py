"""
Project 227. Style transfer for images
Description:
Style transfer is the process of blending the artistic style of one image (like a painting) with the content of another image (like a photo). The result is a new image that looks like the original photo painted in the style of the artwork. This project uses a pre-trained neural style transfer model from PyTorch to apply artistic styles to content images.

About:
What It Does:
This project fuses the visual aesthetics of famous artworks with real-world photos, creating artistic masterpieces with deep learning. It's used in AI art generation, photo filters, and creative design tools, and can be enhanced using faster techniques like Fast Neural Style Transfer for real-time applications.
"""
# Install required packages:
# pip install torch torchvision pillow matplotlib
 
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import matplotlib.pyplot as plt
 
# Use device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Load content and style images
content_path = "content.jpg"  # Replace with your photo
style_path = "style.jpg"      # Replace with an artwork (e.g., Van Gogh painting)
 
def load_image(img_path, max_size=512):
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)
 
content = load_image(content_path)
style = load_image(style_path)
 
# Use VGG19 features for style/content extraction
vgg = vgg19(pretrained=True).features.to(device).eval()
 
# Define layers to use for style/content representations
content_layers = ['21']  # relu4_2
style_layers = ['0', '5', '10', '19', '28']  # relu1_1 to relu5_1
 
# Hook function to extract activations
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features
 
# Calculate Gram matrix for style comparison
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())
 
# Extract features from both images
content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
 
# Initialize the target image as a clone of content
target = content.clone().requires_grad_(True)
 
# Set optimizer and hyperparameters
optimizer = torch.optim.Adam([target], lr=0.003)
style_weight = 1e6
content_weight = 1
 
# Run optimization loop
for i in range(1, 501):
    target_features = get_features(target, vgg, style_layers + content_layers)
    
    # Compute content loss
    content_loss = torch.mean((target_features['21'] - content_features['21']) ** 2)
    
    # Compute style loss
    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        layer_loss = torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_loss
    
    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Step {i}, Loss: {total_loss.item():.4f}")
 
# Display the final stylized image
final_image = target.cpu().clone().detach().squeeze(0)
final_image = final_image.permute(1, 2, 0).clamp(0, 1)
 
plt.figure(figsize=(10, 5))
plt.imshow(final_image)
plt.title("Stylized Output")
plt.axis('off')
plt.show()