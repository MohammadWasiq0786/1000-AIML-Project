"""
Project 903. DDoS Attack Detection

DDoS (Distributed Denial of Service) attacks overwhelm a server with excessive traffic, disrupting availability. This project simulates network flow data and detects potential DDoS patterns using request rate thresholds and IP frequency analysis.

What It Detects:
High-volume requests from a single IP within a short window

Simple frequency-based detection

🛡 In production:

Use time-windowed rate limiting, rolling averages

Apply clustering or anomaly detection on traffic features

Deploy tools like Snort, Suricata, or Cloudflare protection
"""

import pandas as pd
from collections import Counter
 
# Simulated web server request logs (timestamp, source IP)
logs = [
    ("2025-04-11 10:01:01", "192.168.1.5"),
    ("2025-04-11 10:01:02", "192.168.1.5"),
    ("2025-04-11 10:01:03", "192.168.1.5"),
    ("2025-04-11 10:01:04", "10.0.0.8"),
    ("2025-04-11 10:01:05", "192.168.1.5"),
    ("2025-04-11 10:01:06", "192.168.1.5"),
    ("2025-04-11 10:01:07", "192.168.1.5"),
    ("2025-04-11 10:01:08", "172.16.0.3"),
    ("2025-04-11 10:01:09", "192.168.1.5"),
    ("2025-04-11 10:01:10", "192.168.1.5"),
]
 
# Convert to DataFrame
df = pd.DataFrame(logs, columns=["Timestamp", "SourceIP"])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
 
# Count requests per IP
ip_counts = df['SourceIP'].value_counts()
 
# Define threshold (e.g., more than 5 requests in 10 seconds = suspicious)
threshold = 5
suspects = ip_counts[ip_counts > threshold]
 
# Output results
print("DDoS Detection Report:\n")
if not suspects.empty:
    for ip, count in suspects.items():
        print(f"⚠️ Potential DDoS Detected: {ip} made {count} requests in short time")
else:
    print("✅ No DDoS behavior detected.")