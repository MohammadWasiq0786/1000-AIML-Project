"""
Project 454. Drug discovery with AI
Description:
Drug discovery is a complex process involving the identification of new molecules with therapeutic potential. AI accelerates this by predicting drug-target interactions, molecular properties, or bioactivity scores. In this project, we simulate a QSAR (Quantitative Structure-Activity Relationship) model

About:
✅ What It Does:
Converts SMILES strings into molecular fingerprints.

Trains a Random Forest Regressor to predict compound activity.

Simulates a basic QSAR model, commonly used in drug screening.

Extendable to:

Use Graph Neural Networks on molecular graphs

Predict toxicity, solubility, or binding affinity

Integrate with protein docking pipelines

For real applications, use datasets like:

ChEMBL, PubChem BioAssay, or ZINC database
"""




import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
# 1. Simulate a small molecule dataset with SMILES
smiles_data = [
    ('CCO', 0.5),           # ethanol
    ('CCCC', 0.7),          # butane
    ('CC(=O)O', 0.9),       # acetic acid
    ('c1ccccc1', 0.3),      # benzene
    ('CCN(CC)CC', 0.8),     # triethylamine
    ('CC(C)O', 0.6),        # isopropanol
    ('C1=CC=CC=C1O', 0.4),  # phenol
    ('COC(=O)C', 0.7),      # methyl acetate
    ('CN(C)C=O', 0.2),      # dimethylformamide
    ('CC(C)CC(=O)O', 0.9),  # valeric acid
]
 
df = pd.DataFrame(smiles_data, columns=["SMILES", "Activity"])
 
# 2. Convert SMILES to Morgan fingerprints
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp)
 
df['Fingerprint'] = df['SMILES'].apply(smiles_to_fp)
df = df[df['Fingerprint'].notnull()]  # remove invalid SMILES
 
X = np.stack(df['Fingerprint'].values)
y = df['Activity'].values
 
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 5. Evaluate
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2 Score:", r2_score(y_test, y_pred))
 
# 6. Predict on a new compound
new_smiles = 'CC(C)CO'  # isobutanol
new_fp = smiles_to_fp(new_smiles).reshape(1, -1)
predicted_activity = model.predict(new_fp)[0]
print(f"Predicted activity for {new_smiles}: {predicted_activity:.2f}")
