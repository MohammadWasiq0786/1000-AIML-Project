"""
Project 457. Drug-drug interaction prediction
Description:
Drug-Drug Interactions (DDIs) can lead to adverse effects or reduced efficacy when multiple drugs are prescribed. AI models can predict potential interactions between drug pairs using chemical structure embeddings, molecular fingerprints, or graph representations. In this project, we'll implement a simple MLP-based classifier using concatenated SMILES-based fingerprints.

About:
✅ What It Does:
Converts two SMILES drug strings into Morgan fingerprints.

Concatenates them and feeds into a multi-layer perceptron.

Classifies as interaction (1) or no interaction (0).

Can be extended to:

Use graph neural networks over molecular graphs

Predict interaction type/severity

Leverage drug target profiles or pathway features

For real datasets, use:

TWOSIDES

DrugBank

BioSNAP DDI dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import random
 
# 1. Convert SMILES to Morgan fingerprint
def smiles_to_fp(smiles, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)
 
# 2. Simulated DDI dataset (drug1, drug2, interaction label)
pairs = [
    ('CCO', 'CCN', 1),
    ('CCCC', 'CCO', 1),
    ('CC(=O)O', 'CN(C)C=O', 0),
    ('c1ccccc1', 'CCO', 0),
    ('CCOC(=O)C', 'CCN(CC)CC', 1),
    ('C1=CC=CC=C1O', 'CC(C)O', 0),
    ('CC(C)CO', 'CC(=O)O', 1),
    ('CCN', 'CCCN', 0),
]
 
# 3. Create dataset class
class DDIDataset(Dataset):
    def __init__(self, pairs):
        self.samples = []
        for s1, s2, label in pairs:
            fp1 = smiles_to_fp(s1)
            fp2 = smiles_to_fp(s2)
            if fp1 is not None and fp2 is not None:
                x = np.concatenate([fp1, fp2])
                self.samples.append((torch.tensor(x), label))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y
 
# 4. MLP model
class DDIModel(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 2)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)
 
# 5. Setup
dataset = DDIDataset(pairs)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDIModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 6. Training loop
for epoch in range(1, 6):
    model.train()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        preds = output.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Epoch {epoch}, Accuracy: {correct / total:.2f}")