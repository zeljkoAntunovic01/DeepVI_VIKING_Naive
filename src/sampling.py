import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import estimate_sigma, vectorize_nn
from src.data.sinedata import generate_data
from src.losses import sse_loss

def calculate_UUt(model_fn, params_vec, x_train, y_train):
    model_fn_lmbd = lambda p: jnp.squeeze(model_fn(p, x_train))
    J = jax.jacfwd(model_fn_lmbd)(params_vec)
    sigma2_est = estimate_sigma(params_vec, model_fn_lmbd, y_train)

    GGN = (1.0 / sigma2_est) * (J.T @ J)
    D, V = jnp.linalg.eigh(GGN)
    D = jnp.clip(D, a_min=None, a_max=100)
    null_mask = jnp.where(D <= 1e-2, 1.0, 0.0)
    I_p = jnp.eye(GGN.shape[0])

    UUt = I_p - (V @ (1.0 - null_mask)) @ V.T
    return UUt

def sample_theta(key, num_samples, UUt, theta_hat, sigma_ker, sigma_im):
    D = theta_hat.shape[0]
    subkeys = jax.random.split(key, num_samples)
    
    def sample_fn(subkey):
        eps = jax.random.normal(subkey, (D,))
        eps_ker = UUt @ eps
        eps_im = eps - eps_ker
        theta = theta_hat + jnp.exp(sigma_ker) * eps_ker + jnp.exp(sigma_im)* eps_im
        return theta, eps, eps_ker

    thetas, eps_samples, eps_ker_samples = jax.vmap(sample_fn)(subkeys)
    return thetas, eps_samples, eps_ker_samples
