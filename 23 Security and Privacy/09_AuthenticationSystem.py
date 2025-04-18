"""
Project 889. Authentication System

An authentication system verifies user identity to grant access to secure systems. In this project, we simulate a multi-factor authentication (MFA) mechanism using a simple logic that checks username-password match and a second authentication factor (e.g., OTP or device).

Features:
Password Hashing: Ensures passwords are not stored in plain text.

OTP-based Second Factor: Adds an extra layer of security.

Result Feedback: Lets the user know which step failed.

For production:

Integrate with a database and secure OTP service (e.g., TOTP via Google Authenticator)

Add retry limits and IP logging

Use OAuth or biometric verification in advanced setups
"""


import hashlib
import random
 
# Simulated user database (username: hashed_password)
users = {
    'alice': hashlib.sha256('alice123'.encode()).hexdigest(),
    'bob': hashlib.sha256('bob456'.encode()).hexdigest(),
}
 
# Function to verify credentials
def verify_credentials(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return users.get(username) == hashed
 
# Simulated OTP system (for demo, we fix OTP for simplicity)
def generate_otp():
    return str(random.randint(100000, 999999))
 
# Authenticate user
def authenticate(username, password, otp_input, actual_otp):
    if verify_credentials(username, password):
        print("✅ Password verified.")
        if otp_input == actual_otp:
            print("✅ OTP verified. Access granted.")
            return True
        else:
            print("❌ Invalid OTP. Access denied.")
    else:
        print("❌ Invalid username or password.")
    return False
 
# Simulated login attempt
username = 'alice'
password = 'alice123'
actual_otp = generate_otp()
print(f"(Simulated OTP sent to device: {actual_otp})")
 
# User inputs
otp_input = input("Enter OTP: ")
 
# Authenticate user with both factors
authenticate(username, password, otp_input, actual_otp)