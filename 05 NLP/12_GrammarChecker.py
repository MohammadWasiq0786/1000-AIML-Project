"""
Project 172. Grammar checker implementation
Description:
A grammar checker detects grammatical errors and suggests corrections to improve sentence structure, verb tense, punctuation, and more. In this project, we build a simple grammar correction tool using language_tool_python, a Python wrapper for LanguageTool, an open-source grammar checker.

About:
🧠 What This Project Demonstrates:
Uses LanguageTool via Python to check and correct grammar mistakes

Provides automatic corrections and detailed feedback

Works on common errors like subject-verb agreement, punctuation, and word misuse
"""

# Install if not already: pip install language-tool-python
 
import language_tool_python
 
# Initialize the grammar checking tool (English)
tool = language_tool_python.LanguageTool('en-US')
 
# Example sentences with grammar issues
sentences = [
    "She go to the store every day.",
    "I has a apple in my bag.",
    "Their going too the movies tonite.",
    "This are bad example of english.",
]
 
print("🧠 Grammar Checker Results:\n")
 
for sentence in sentences:
    # Check for grammar errors
    matches = tool.check(sentence)
    corrected = language_tool_python.utils.correct(sentence, matches)
    
    print(f"❌ Original:  {sentence}")
    print(f"✅ Corrected: {corrected}")
    print(f"🔧 Issues found: {len(matches)}\n")