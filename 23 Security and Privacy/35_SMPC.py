"""
Project 915. Secure Multi-Party Computation (SMPC)

Secure Multi-Party Computation (SMPC) allows multiple parties to collaboratively compute a function over their inputs without revealing their inputs to each other. In this project, we simulate SMPC using secret sharing, where data is split into random shares that individually reveal nothing.

What This Shows:
No party sees the actual inputs of others.

All parties collaborate to compute the total securely.

Works for basic operations like sum, average, dot product.

🔒 Real-world SMPC frameworks:

CrypTen (PyTorch), MP-SPDZ, FRESCO

Used in privacy-preserving data collaboration, joint ML training, private voting
"""

import random
 
# Simulated private values from three parties
party1_input = 30
party2_input = 45
party3_input = 25
 
# Step 1: Split each input into 3 random shares
def split_secret(secret, n=3):
    shares = [random.randint(0, 100) for _ in range(n - 1)]
    final_share = secret - sum(shares)
    shares.append(final_share)
    return shares
 
# Each party creates their shares
shares1 = split_secret(party1_input)
shares2 = split_secret(party2_input)
shares3 = split_secret(party3_input)
 
# Step 2: Distribute shares (simulate by transposing the share lists)
all_shares = list(zip(shares1, shares2, shares3))
 
# Step 3: Each party computes local sum of received shares
partial_sums = [sum(share_group) for share_group in all_shares]
 
# Step 4: Combine partial sums to get the final result
final_sum = sum(partial_sums)
 
print("Secure Multi-Party Computation (Addition):")
print(f"Party 1 input: {party1_input}")
print(f"Party 2 input: {party2_input}")
print(f"Party 3 input: {party3_input}")
print(f"\nComputed Secure Sum: {final_sum} ✅")