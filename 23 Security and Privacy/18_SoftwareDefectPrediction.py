"""
Project 898. Software Defect Prediction

Software defect prediction helps identify buggy code components before deployment, using historical metrics like code complexity, size, and previous defect labels. In this project, we simulate a dataset of software modules and train a classifier to predict whether a module is likely to contain a defect.

Why It Works:
Uses metrics extracted from static code analysis tools (e.g., SonarQube, Radon).

Predicts whether a module is defect-prone based on complexity and structure.

🔍 In production:

Extend with real code metrics from version control or CI/CD pipelines.

Use models like logistic regression, XGBoost, or neural nets with embedding from source code.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated software module dataset
data = {
    'LinesOfCode': [120, 450, 90, 600, 150, 700, 100, 550],
    'CyclomaticComplexity': [5, 15, 3, 20, 7, 25, 4, 18],
    'NumFunctions': [3, 10, 2, 12, 4, 15, 3, 13],
    'Defect': [0, 1, 0, 1, 0, 1, 0, 1]  # 1 = defect-prone, 0 = clean
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Defect', axis=1)
y = df['Defect']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Software Defect Prediction Report:")
print(classification_report(y_test, y_pred))
 
# Predict on new code module
new_module = pd.DataFrame([{
    'LinesOfCode': 500,
    'CyclomaticComplexity': 19,
    'NumFunctions': 11
}])
defect_risk = model.predict_proba(new_module)[0][1]
print(f"\nPredicted Defect Risk: {defect_risk:.2%}")