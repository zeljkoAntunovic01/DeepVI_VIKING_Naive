import jax
import pickle
from utils import vectorize_nn
import matplotlib.pyplot as plt
import jax.numpy as jnp

N = 150
SEED = 42

prior_key, post_key = jax.random.split(jax.random.PRNGKey(SEED), num=2)

# Generate prior variances
prior_vec = jnp.clip(jax.random.normal(prior_key, (550,)) ** 2, 0.1, 10.0)

# Get min and max
prior_max = prior_vec.max()
prior_min = prior_vec.min()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(prior_vec, bins=50)
plt.title("Prior Variances")
plt.yscale("log")

# Annotate max and min
plt.text(prior_max, 1, f"Max: {prior_max:.2f}", ha="right", va="bottom", fontsize=10, color="red")
plt.text(prior_min, 1, f"Min: {prior_min:.2e}", ha="left", va="bottom", fontsize=10, color="blue")

plt.xlabel("Variance")
plt.ylabel("Frequency (log scale)")
plt.tight_layout()
plt.show()

# Also print values to console
print("Max prior variance:", prior_max)
print("Min prior variance:", prior_min)
