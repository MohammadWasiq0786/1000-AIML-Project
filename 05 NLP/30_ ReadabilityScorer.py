"""
Project 190. Readability scorer
Description:
A Readability Scorer evaluates how easy or difficult a text is to read. It’s especially useful in education, publishing, and accessibility tools. This project calculates various readability metrics like Flesch Reading Ease, Flesch-Kincaid Grade Level, and Gunning Fog Index using the textstat library.

About:
🧠 What This Project Demonstrates:
Analyzes text for reading difficulty and complexity

Provides multiple standardized readability metrics

Helps tailor content for different audience literacy levels
"""

# Install if not already: pip install textstat
 
import textstat
 
# Sample text (can be user input or from a document)
sample_text = """
Artificial Intelligence is transforming the way industries operate, 
enabling automation, decision-making, and customer personalization at scale.
However, its integration demands ethical considerations, transparency, and oversight.
"""
 
# Calculate different readability scores
flesch_score = textstat.flesch_reading_ease(sample_text)
grade_level = textstat.flesch_kincaid_grade(sample_text)
gunning_fog = textstat.gunning_fog(sample_text)
smog_index = textstat.smog_index(sample_text)
automated_readability = textstat.automated_readability_index(sample_text)
 
# Display the results
print("🧠 Readability Scores:\n")
print(f"📊 Flesch Reading Ease: {flesch_score:.2f} (Higher = easier)")
print(f"🎓 Flesch-Kincaid Grade Level: {grade_level:.2f}")
print(f"🌫️ Gunning Fog Index: {gunning_fog:.2f}")
print(f"🔢 SMOG Index: {smog_index:.2f}")
print(f"📈 Automated Readability Index: {automated_readability:.2f}")