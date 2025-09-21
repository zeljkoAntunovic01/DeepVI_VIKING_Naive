import pickle
import jax.numpy as jnp

# Load model, extract params...
with open("./checkpoints/sine_VIKING_ALPHA_exact.pickle", "rb") as f:
    checkpoint = pickle.load(f)

elbo_stats = checkpoint["elbo_params"]

print(f"Sigma_ker = {jnp.exp(elbo_stats["sigma_ker"])}")
print(f"Sigma_im = {jnp.exp(elbo_stats["sigma_im"])}")