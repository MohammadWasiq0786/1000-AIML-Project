"""
Project 6. Naive Bayes classifier for spam detection
Description:
Naive Bayes is a probabilistic classifier based on Bayes’ Theorem with an assumption of feature independence. It's especially effective for spam detection, thanks to its ability to handle high-dimensional text data. In this project, we’ll build a spam filter using a bag-of-words model and train a Naive Bayes classifier with scikit-learn.

About:
This model uses the Multinomial Naive Bayes algorithm with a bag-of-words representation, making it suitable for large-scale spam filtering tasks.
"""

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
 
# Sample dataset: (email message, label)
# Label: 1 = Spam, 0 = Ham (not spam)
data = [
    ("Congratulations, you've won a $1000 Walmart gift card. Click to claim now!", 1),
    ("Reminder: Your appointment is scheduled for 10 AM tomorrow.", 0),
    ("URGENT: You have been selected for a prize. Act fast!", 1),
    ("Hi, can we reschedule our meeting?", 0),
    ("Earn money from home without doing anything!", 1),
    ("Your Amazon order has been shipped.", 0),
    ("Claim your free vacation now by clicking this link!", 1),
    ("Lunch at 1 PM?", 0),
    ("You are pre-approved for a $5000 loan. No credit check!", 1),
    ("Can you send me the report by EOD?", 0)
]
 
# Split the dataset into texts and labels
texts, labels = zip(*data)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
 
# Convert text into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
 
# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)
 
# Predict on the test set
y_pred = classifier.predict(X_test_vec)
 
# Print evaluation results
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
 
# Test with a custom email message
custom_email = ["Free entry in a contest to win $10000 cash!"]
custom_vec = vectorizer.transform(custom_email)
prediction = classifier.predict(custom_vec)

print("\nCustom Email Prediction:")
print(f"'{custom_email[0]}' is classified as:", "Spam" if prediction[0] == 1 else "Ham")