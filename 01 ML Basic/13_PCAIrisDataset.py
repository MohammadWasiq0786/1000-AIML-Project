"""
Project 13. Principal component analysis implementation
Description:
Principal Component Analysis (PCA) is a statistical technique that reduces the dimensionality of data while preserving as much variance as possible. It helps with visualization, noise reduction, and improving performance of other algorithms. In this project, we’ll use PCA to reduce the Iris dataset from 4D to 2D for visualization.

About:
What PCA does:

Identifies the directions (principal components) of maximum variance

Projects data into lower dimensions with minimal information loss

Helps in visualization and preprocessing for machine learning
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # 4 features
y = iris.target      # 3 classes: setosa, versicolor, virginica
labels = iris.target_names
 
# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
# Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", sum(pca.explained_variance_ratio_))
 
# Plot the 2D PCA result
plt.figure(figsize=(8, 6))
for i in range(len(labels)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=labels[i])
    
plt.title("PCA on Iris Dataset (2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()