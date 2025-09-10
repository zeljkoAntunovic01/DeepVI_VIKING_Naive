import pickle

# Load model, extract params...
with open("./checkpoints/sine_VIKING.pickle", "rb") as f:
    checkpoint = pickle.load(f)

elbo_stats = checkpoint["elbo_params"]

print(f"Sigma_ker = {elbo_stats["sigma_ker"]}")
print(f"Sigma_im = {elbo_stats["sigma_im"]}")