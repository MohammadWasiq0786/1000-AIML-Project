"""
Project 38. t-SNE visualization tool
Description:
t-SNE (t-distributed Stochastic Neighbor Embedding) is a powerful dimensionality reduction technique particularly suited for visualizing high-dimensional data in 2D or 3D. It preserves local structure and clusters similar samples together. In this project, we create a t-SNE visualization tool using the Iris dataset, and display its 2D embedding by class.

About:
🎯 What This Tool Demonstrates:
Transforms 4D data into 2D using t-SNE

Reveals natural clusters and class separations

Provides interpretable visual feedback on data structure
"""

# Import required libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import pandas as pd
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
 
# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)
 
# Create a DataFrame for easy plotting
df_tsne = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
df_tsne['Target'] = y
 
# Plot t-SNE result
plt.figure(figsize=(8, 6))
for i, label in enumerate(target_names):
    subset = df_tsne[df_tsne['Target'] == i]
    plt.scatter(subset["Component 1"], subset["Component 2"], label=label, s=60)
 
plt.title("t-SNE Visualization of Iris Dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()