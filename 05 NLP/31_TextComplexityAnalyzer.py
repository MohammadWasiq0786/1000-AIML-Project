"""
Project 191. Text complexity analyzer
Description:
A Text Complexity Analyzer evaluates how complex a piece of text is using multiple linguistic features such as sentence length, word length, lexical diversity, and syntactic depth. It’s especially useful in education, content creation, and AI tutoring systems to assess whether content is appropriate for a specific reading level.

About:
🧠 What This Project Demonstrates:
Analyzes surface-level and deep complexity features

Measures lexical richness, sentence structure, and readability

Provides a composite view of text difficulty
"""

# Install if not already: pip install nltk textstat
 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat
 
nltk.download('punkt')
nltk.download('stopwords')
 
# Sample text for analysis
text = """
The advancement of artificial intelligence and machine learning has significantly impacted modern industries. 
With the ability to analyze massive amounts of data, these technologies empower organizations to make informed decisions, 
automate routine tasks, and enhance customer engagement through personalized experiences.
"""
 
# Tokenization
words = word_tokenize(text)
sentences = sent_tokenize(text)
stop_words = set(stopwords.words('english'))
 
# Complexity features
avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
avg_word_length = sum(len(word) for word in words if word.isalpha()) / len(words)
lexical_diversity = len(set(words)) / len(words)
flesch_reading_ease = textstat.flesch_reading_ease(text)
 
# Display results
print("🧠 Text Complexity Analysis:\n")
print(f"📝 Number of Sentences: {len(sentences)}")
print(f"🔤 Number of Words: {len(words)}")
print(f"📏 Average Sentence Length: {avg_sentence_length:.2f} words")
print(f"🔠 Average Word Length: {avg_word_length:.2f} characters")
print(f"🧬 Lexical Diversity: {lexical_diversity:.2f} (0-1 scale)")
print(f"📚 Flesch Reading Ease Score: {flesch_reading_ease:.2f} (Higher = Easier to Read)")