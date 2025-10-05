import pickle
import jax.numpy as jnp

# Load model, extract params...
with open("./checkpoints/sine_VIKING_GGN.pickle", "rb") as f:
    checkpoint = pickle.load(f)

elbo_stats = checkpoint["elbo_params"]

print(f"Log sigma_ker = {elbo_stats['sigma_ker']}")
print(f"Log sigma_im = {elbo_stats['sigma_im']}")
print(f"Sigma_ker = {jnp.exp(elbo_stats["sigma_ker"])}")
print(f"Sigma_im = {jnp.exp(elbo_stats["sigma_im"])}")