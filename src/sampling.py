import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import estimate_sigma, vectorize_nn
from src.data.sinedata import generate_data
from src.losses import sse_loss

def compute_J(params_vec, model_fn, x_train, y_train):
    """
    Compute per-sample Jacobian of the loss wrt parameters.
    J has shape (N, D).
    """
    def per_sample_loss(p, x, y):
        y_pred = model_fn(p, x)
        return jnp.square(y_pred - y).sum()  # SSE per sample
    
    lmbd = lambda p: jax.vmap(per_sample_loss, in_axes=(None, 0, 0))(p, x_train, y_train)
    # Vectorize over data points
    J = jax.jacfwd(lmbd)(params_vec)  # (N, D)
    return J

#----------------------------------------------------------
# First method: Using GGN approximation
#----------------------------------------------------------
def calculate_UUt(model_fn, params_vec, x_train, y_train):
    model_fn_lmbd = lambda p: jnp.squeeze(model_fn(p, x_train))
    J = compute_J(params_vec, model_fn, x_train, y_train)  # (N, D)
    sigma2_est = estimate_sigma(params_vec, model_fn_lmbd, y_train)

    GGN = (1.0 / sigma2_est) * (J.T @ J)
    D, V = jnp.linalg.eigh(GGN)
    D = jnp.clip(D, a_min=None, a_max=100)
    null_mask = jnp.where(D <= 1e-2, 1.0, 0.0)
    I_p = jnp.eye(GGN.shape[0])

    UUt = I_p - (V @ (1.0 - null_mask)) @ V.T
    return UUt

def calculate_UUt_svd(model_fn, params_vec, x_train, y_train, tol=1e-2):
    # Jacobian: shape (N, D)
    J = compute_J(params_vec, model_fn, x_train, y_train)  # (N, D)
    D = J.shape[1]

    # Full SVD (J = U Σ V^T)
    U, S, VT = jnp.linalg.svd(J, full_matrices=True)  # VT: (D, D), S: (min(N, D),)

    # Pad singular values up to D (if N < D)
    S_full = jnp.pad(S, (0, D - S.shape[0]), constant_values=0.0)

    # Identify "nullspace" dimensions (dynamic mask)
    null_mask = S_full <= tol  # shape (D,)
    rank_ker = jnp.sum(null_mask)

    # Build a diagonal mask (float) instead of boolean indexing
    diag_mask = null_mask.astype(J.dtype)  # (D,)

    # Projection: UUt = V diag(mask) V^T
    V = VT.T  # (D, D)
    UUt = V @ (diag_mask * V.T)  # broadcasting applies mask to rows/cols

    return UUt, rank_ker

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

#----------------------------------------------------------
# Second method: Using Jacobians and solving a system
#----------------------------------------------------------
def project_to_kernel(J, eps, ridge=1e-4):
    JJt = J @ J.T + ridge * jnp.eye(J.shape[0])
    b = J @ eps
    lam = jnp.linalg.solve(JJt, b)   # much more stable than lstsq
    correction = J.T @ lam
    return eps - correction

def sample_theta_exact(key, num_samples, J, theta_hat, sigma_ker, sigma_im):
    """
    Sample posterior parameters using exact kernel projection.
    """
    D = theta_hat.shape[0]
    subkeys = jax.random.split(key, num_samples)

    def sample_fn(subkey):
        eps = jax.random.normal(subkey, (D,))
        eps_ker = project_to_kernel(J, eps)
        eps_im = eps - eps_ker
        theta = theta_hat + jnp.exp(sigma_ker) * eps_ker + jnp.exp(sigma_im) * eps_im
        dot_product = jnp.dot(eps_ker, eps_im)
        return theta, eps, eps_ker, dot_product

    thetas, eps_samples, eps_ker_samples, dot_products = jax.vmap(sample_fn)(subkeys)
    return thetas, eps_samples, eps_ker_samples, dot_products
