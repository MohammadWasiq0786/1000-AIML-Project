"""
Project 914. Homomorphic Encryption Demos

Homomorphic Encryption (HE) allows computations on encrypted data without decrypting it—preserving privacy during computation. In this project, we demonstrate encrypted addition and multiplication using the PySEAL or TenSEAL library.

📌 For this example, we’ll use TenSEAL (easy to install via pip install tenseal).


Why It’s Powerful:
You can perform operations without revealing raw data

Used in secure cloud computation, medical analytics, financial analysis

🔐 Real-world enhancements:

Use BFV scheme for exact integer math

Combine with federated learning or multi-party computation

Protect model parameters and user queries simultaneously
"""

import tenseal as ts
 
# Create TenSEAL context with CKKS scheme (supports real numbers)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40
 
# Step 1: Encrypt two numbers
x = 10.0
y = 5.0
 
enc_x = ts.ckks_vector(context, [x])
enc_y = ts.ckks_vector(context, [y])
 
# Step 2: Perform computations on encrypted data
enc_sum = enc_x + enc_y
enc_product = enc_x * enc_y
 
# Step 3: Decrypt results
decrypted_sum = enc_sum.decrypt()[0]
decrypted_product = enc_product.decrypt()[0]
 
# Output
print(f"Encrypted Input: x = {x}, y = {y}")
print(f"Decrypted Sum (x + y): {decrypted_sum:.2f}")
print(f"Decrypted Product (x * y): {decrypted_product:.2f}")