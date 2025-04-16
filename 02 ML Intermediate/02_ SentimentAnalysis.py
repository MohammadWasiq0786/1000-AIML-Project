"""
Project 42. Sentiment analysis for movie reviews
Description:
Sentiment analysis determines whether text expresses positive, negative, or neutral emotion. In this project, we build a binary sentiment classifier for movie reviews using the IMDb dataset (or custom input). We’ll apply TF-IDF vectorization and train a Logistic Regression model using scikit-learn.

About:
🧠 What This Project Demonstrates:
Uses TF-IDF vectorization for converting text into numeric form

Trains a logistic regression classifier for binary sentiment detection

Visualizes results using confusion matrix and detailed performance report
"""

# Install if needed:
# pip install sklearn
 
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Option 1: Use your own dataset
# Option 2: Use a small custom list (for simplicity here)
 
reviews = [
    "I absolutely loved this movie. The story was fantastic and acting was brilliant.",
    "This was a waste of time. Horrible plot and bad direction.",
    "Great film! Will definitely watch again.",
    "Terrible experience. I walked out halfway.",
    "Not bad, but could have been better.",
    "Best movie I’ve seen this year!",
    "Awful. Just awful. Please don’t watch.",
    "A masterpiece. Well done to the cast.",
    "I didn’t like the movie. It was boring.",
    "Amazing movie! Beautiful visuals and emotional story."
]
 
labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
 
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42)
 
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.title("Confusion Matrix - Sentiment Analysis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
