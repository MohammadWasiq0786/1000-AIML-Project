"""
Project 902. Security Log Analysis

Security log analysis helps detect suspicious or unauthorized activities by analyzing system, network, or application logs. In this project, we simulate a simple log file and use Python to flag login failures, unauthorized access attempts, and unusual activity times.

What It Detects:
Authentication failures (login failed)

Access during odd hours (e.g., 2 AM, 3 AM)

Use of suspicious or undefined usernames

🔍 In real systems:

Use SIEM platforms (e.g., Splunk, ELK Stack)

Apply regular expressions for deep parsing
"""

import pandas as pd
import re
 
# Simulated log entries
logs = [
    "2025-04-11 09:01:05 | INFO | user=alice | login success",
    "2025-04-11 09:15:33 | ERROR | user=bob | login failed",
    "2025-04-11 02:45:00 | INFO | user=admin | access system settings",
    "2025-04-11 13:22:10 | INFO | user=alice | download confidential.pdf",
    "2025-04-11 03:00:12 | ERROR | user=guest | login failed",
    "2025-04-11 23:50:00 | INFO | user=unknown | login success",
]
 
# Convert log data into structured format
data = []
for log in logs:
    parts = re.split(r'\s\|\s', log)
    timestamp, level, user_info, message = parts
    user = re.search(r'user=(\w+)', user_info).group(1)
    data.append({'Timestamp': timestamp, 'Level': level, 'User': user, 'Message': message})
 
df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
 
# Rule-based anomaly detection
print("Security Log Analysis Report:\n")
 
# 1. Failed logins
failed_logins = df[df['Message'].str.contains('login failed')]
print("🔐 Failed Login Attempts:")
print(failed_logins[['Timestamp', 'User', 'Message']], end="\n\n")
 
# 2. Unusual access times (e.g., between 00:00 and 05:00)
odd_hours = df[df['Hour'].between(0, 5)]
print("🌙 Activity at Unusual Hours:")
print(odd_hours[['Timestamp', 'User', 'Message']], end="\n\n")
 
# 3. Suspicious or unknown users
suspicious_users = df[df['User'].isin(['guest', 'unknown'])]
print("⚠️ Suspicious Users:")
print(suspicious_users[['Timestamp', 'User', 'Message']])