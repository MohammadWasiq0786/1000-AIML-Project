"""
Project 464. Mental health monitoring
Description:
Mental health monitoring uses AI to detect signs of stress, anxiety, or depression through text input, voice tone, or behavioral patterns (e.g., sleep, phone usage). In this project, we simulate a text-based mood analysis system using sentiment classification to monitor users’ emotional states over time.

✅ What It Does:
Takes user journal input and detects mood/emotion using a sentiment classifier.

Logs timestamped entries and confidence scores.

Can be extended to:

Use emotion classifiers (e.g., anger, fear, joy)

Integrate with voice analysis, sleep tracking, or chatbots

Trigger alerts or suggestions when negative patterns are detected

For real-world deployment:

Integrate with data from journaling apps, chatbots, or social media

Use models trained on mental health datasets (e.g., CLPsych, DAIC-WOZ)
"""

from transformers import pipeline
import datetime
 
# 1. Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")
 
# 2. Store mood log entries
mood_log = []
 
# 3. Start monitoring loop
print("Welcome to MindTrack. Type your daily journal entry. Type 'exit' to stop.\n")
 
while True:
    entry = input("Your Entry: ")
    if entry.lower() == "exit":
        break
 
    # Analyze sentiment
    result = sentiment_model(entry)[0]
    mood = result["label"]
    score = result["score"]
    
    # Save entry with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mood_log.append({"time": timestamp, "entry": entry, "mood": mood, "confidence": round(score, 2)})
 
    # Feedback to user
    print(f"Detected mood: {mood} (Confidence: {score:.2f})")
    print("-" * 50)
 
# 4. Summary report
print("\nMood Monitoring Summary:")
for log in mood_log:
    print(f"[{log['time']}] Mood: {log['mood']} | Entry: {log['entry'][:50]}...")