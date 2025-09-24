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
    
    # Vectorize over data points
    per_sample_grad = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))
    J = per_sample_grad(params_vec, x_train, y_train)  # (N, D)
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

def calculate_UUt_svd(model_fn, params_vec, x_train, y_train, rtol=1e-4):
    # Jacobian: shape (N, D)
    J = compute_J(params_vec, model_fn, x_train, y_train)  # (N, D)
    D = J.shape[1]

    # Full SVD (J = U Σ V^T)
    U, S, VT = jnp.linalg.svd(J, full_matrices=False)  # VT: (D, D)

    # Determine rank (non-negligible singular values)
    tol = rtol * jnp.max(S)
    r = jnp.sum(S > tol)

    # Kernel basis is last D - r columns of V^T
    V_null = VT[r:].T  # (D, D - r)

    # Projection onto kernel subspace
    UUt = V_null @ V_null.T
    return UUt, D - r

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
def project_to_kernel(J, eps):
    """
    Project eps onto ker(J) using Eq. (15):
    eps_ker = eps - J^T (J J^T)^{-1} J eps
    """
    b = J @ eps                   # (N,)
    JJt = J @ J.T                  # (N, N)
    lam = jnp.linalg.solve(JJt, b) # (N,)
    correction = J.T @ lam         # (D,)
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
        return theta, eps, eps_ker

    thetas, eps_samples, eps_ker_samples = jax.vmap(sample_fn)(subkeys)
    return thetas, eps_samples, eps_ker_samples
