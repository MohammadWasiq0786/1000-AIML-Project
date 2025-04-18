"""
Project 877: Education Resource Allocation
Description
Education resource allocation helps ensure that regions with the highest need receive adequate funding, teachers, and infrastructure. In this project, we simulate demographic, infrastructure, and performance data and build a regression model to predict the optimal resource allocation for schools (e.g., teacher-student ratio, funding, infrastructure improvements).

✅ This model can be used for:

Education funding optimization

School improvement planning

Government education policies
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate education-related data
np.random.seed(42)
n_samples = 1000
 
student_population = np.random.normal(500, 150, n_samples)     # number of students
teacher_population = np.random.normal(30, 10, n_samples)        # number of teachers
school_funding = np.random.normal(1000000, 300000, n_samples)  # USD
classroom_space = np.random.uniform(0.4, 1.0, n_samples)        # % of ideal space
internet_access_score = np.random.uniform(0, 1, n_samples)     # 0–1 scale
 
# Simulate resource allocation needs (funding per student, teacher-student ratio, infrastructure)
teacher_student_ratio = teacher_population / student_population
funding_per_student = school_funding / student_population
classroom_adequacy = (classroom_space * 100) - 40  # how much classroom space deviates from ideal
tech_infrastructure_need = (1 - internet_access_score) * 50  # scale of digital infrastructure improvement needed
 
# Combine features
X = np.stack([student_population, teacher_population, school_funding, classroom_space, internet_access_score], axis=1)
y = np.stack([funding_per_student, teacher_student_ratio, classroom_adequacy, tech_infrastructure_need], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model for resource allocation
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)  # 4 outputs: funding/student, teacher-student ratio, classroom adequacy, tech need
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Education Resource Allocation MAE: {mae:.2f}")
 
# Predict for 5 sample schools
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"\nSchool {i+1} Resource Allocation:")
    print(f"  💸 Funding per Student: ${preds[i][0]:,.2f}")
    print(f"  👩‍🏫 Teacher-Student Ratio: {preds[i][1]:.2f}")
    print(f"  🏫 Classroom Adequacy: {preds[i][2]:.2f}%")
    print(f"  💻 Tech Infrastructure Need: {preds[i][3]:.2f}")