"""
Project 896. Password Strength Evaluation

A password strength evaluation system assesses how secure a password is based on its length, complexity, and entropy. In this project, we simulate a basic password strength meter that classifies passwords as Weak, Moderate, or Strong.

What It Checks:
Length: Longer passwords are harder to brute-force.

Character diversity: Encourages use of uppercase, lowercase, digits, symbols.

Score-based logic: Adjustable thresholds for real-world applications.

🔐 Advanced versions may include:

Entropy calculation

Blacklist check (common passwords)

Zxcvbn scoring (as used by Dropbox)
"""

import re
 
# Function to evaluate password strength
def evaluate_password(password):
    length = len(password)
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_symbol = bool(re.search(r'\W', password))  # Non-alphanumeric
    score = sum([has_upper, has_lower, has_digit, has_symbol])
 
    # Rule-based classification
    if length >= 12 and score == 4:
        return 'Strong 💪'
    elif length >= 8 and score >= 3:
        return 'Moderate ⚠️'
    else:
        return 'Weak ❌'
 
# Test passwords
passwords = [
    "password",              # weak
    "Pass1234",              # moderate
    "Str0ng!Pass2024",       # strong
    "letmein",               # weak
    "Admin@987",             # moderate
    "aB1@"                   # weak (short)
]
 
print("Password Strength Evaluation:")
for pwd in passwords:
    print(f"{pwd}: {evaluate_password(pwd)}")