"""
Project 904. Bot Detection System

A bot detection system identifies automated scripts or bots masquerading as real users—often used for spamming, scraping, or brute-force attacks. In this project, we simulate user session behavior and flag likely bots using rule-based heuristics and simple ML features.

Detection Logic:
High click rate: Bots often click faster than humans.

Short session duration: Bots complete tasks quickly.

Little or no mouse movement: Bots don’t move cursors like humans.

🤖 For real-world accuracy:

Use features like scroll depth, typing cadence, user-agent patterns

Apply behavioral ML models or deep learning with session replay

Combine with CAPTCHA challenges or bot fingerprinting libraries (e.g., BotD, Cloudflare)
"""

import pandas as pd
 
# Simulated session data: one row per user session
data = {
    'SessionID': [1, 2, 3, 4, 5, 6],
    'ClicksPerSecond': [0.5, 1.2, 10.5, 0.7, 12.0, 0.4],
    'PagesVisited': [5, 6, 1, 4, 2, 7],
    'SessionDuration': [300, 250, 20, 270, 15, 310],  # in seconds
    'MouseMovements': [200, 180, 0, 190, 1, 210]       # number of movements
}
 
df = pd.DataFrame(data)
 
# Simple rule-based bot detection
def is_bot(row):
    if row['ClicksPerSecond'] > 5 or row['SessionDuration'] < 30 or row['MouseMovements'] < 5:
        return 1  # bot
    return 0      # human
 
df['IsBot'] = df.apply(is_bot, axis=1)
 
# Show detected bots
print("Bot Detection Report:\n")
print(df[['SessionID', 'ClicksPerSecond', 'SessionDuration', 'MouseMovements', 'IsBot']])