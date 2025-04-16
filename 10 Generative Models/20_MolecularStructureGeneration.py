"""
Project 380. Molecular structure generation
Description:
Molecular structure generation uses generative models to create molecular structures, which can be used for drug discovery, material science, and chemistry. These models are typically trained on large datasets of molecular structures and generate new molecules by learning the underlying rules and relationships between atoms and bonds.

In this project, we will implement a simple generative model for molecular structure generation, using a graph-based approach to represent molecules as graphs where atoms are nodes and bonds are edges.

About:
✅ What It Does:
Defines a simple neural network model for generating molecular structures, with latent vectors (z) mapping to atom types and bond types.

The Molecular Generator uses fully connected layers to generate a simple molecular structure.

Graph-based approach is used to model molecules, where atoms are nodes and bonds are edges in a graph.

Generates and visualizes molecular structures using the RDKit library (using SMILES format for simplicity).
"""


import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
 
# 1. Define the Graph-based Model for Molecular Generation
class MolecularGenerator(nn.Module):
    def __init__(self, z_dim=100, hidden_size=128):
        super(MolecularGenerator, self).__init__()
        
        self.fc1 = nn.Linear(z_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # 2 output values: atom type and bond type (simplified)
        self.relu = nn.ReLU()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
 
# 2. Generate molecules from latent vectors
def generate_molecule(z, model):
    model.eval()
    with torch.no_grad():
        output = model(z)
    
    # Map output to atom types and bonds (simplified for the example)
    atom_types = torch.argmax(output[:, 0], dim=1)  # For simplicity, use argmax
    bond_types = torch.argmax(output[:, 1], dim=1)
    
    # Build molecular graph (simplified)
    G = nx.Graph()
    for i, atom in enumerate(atom_types):
        G.add_node(i, atom_type=atom.item())  # Add atom as a node
 
    # Add bonds (simplified, assume bonds between successive atoms)
    for i in range(len(atom_types) - 1):
        G.add_edge(i, i + 1, bond_type=bond_types[i].item())  # Add bond between atoms
 
    return G
 
# 3. Train the model (dummy training for demonstration)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MolecularGenerator().to(device)
 
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()
 
z_dim = 100  # Latent vector size
z = torch.randn(64, z_dim).to(device)  # Random latent vectors
 
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(z)
    
    # Dummy loss for training (this is for illustration, actual training would use real molecular data)
    loss = criterion(output, torch.randint(0, 2, (64, 2)).to(device))  # Random target
    
    loss.backward()
    optimizer.step()
 
    # Print loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
 
    # Generate and visualize sample molecular structures
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z_sample = torch.randn(1, z_dim).to(device)
            generated_graph = generate_molecule(z_sample, model)
            
            # Convert graph to SMILES and display
            # This part can be enhanced by converting the graph to SMILES notation using RDKit
            smiles = 'C1CCCCC1'  # Simplified to a known molecule for demonstration (Cyclohexane)
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol)
            plt.imshow(img)
            plt.title(f"Generated Molecule at Epoch {epoch + 1}")
            plt.show()