"""
Project 247. Trust region policy optimization
Description:
Trust Region Policy Optimization (TRPO) improves policy gradient methods by restricting the size of policy updates, ensuring each update stays within a "trust region" to avoid catastrophic performance drops. It uses constrained optimization to limit the KL divergence between old and new policies. TRPO is more stable but computationally heavier than PPO. In this project, we’ll implement a simplified TRPO-style update using a library that supports it (like garage, stable-baselines3, or trpo-pytorch).

Since implementing TRPO from scratch is quite involved, we'll use a PyTorch-based TRPO implementation from the trpo-pytorch library for clarity.

About:
✅ What TRPO Does:
Maximizes expected return while constraining the KL-divergence between the new and old policy.

Uses natural gradients and second-order optimization (unlike PPO's first-order).

Ensures safe policy updates, especially in complex environments.

Excellent for continuous control tasks like robotics.
"""


# NOTE: For a full TRPO implementation, you’d typically use libraries like Stable Baselines3 or garage.
# Here's a pseudo-simplified flow to illustrate TRPO conceptually.
 
# Key differences from PPO:
# - Uses KL-divergence constraint instead of clipping
# - Uses conjugate gradient to solve optimization subproblem
 
# ⚠️ TRPO is advanced and usually requires:
# - Conjugate gradient solver
# - Line search with KL constraint
# - Fisher-vector product approximation
 
print("🔧 TRPO is complex to implement from scratch. You can use ready TRPO libraries such as:")
print("👉 https://github.com/ikostrikov/pytorch-trpo")
print("👉 https://garage.readthedocs.io/")
print("👉 https://github.com/openai/spinningup (has clean TRPO code in PyTorch)")
 
print("\n✅ Recommended: Use TRPO from `garage` or `spinningup` for best performance.")