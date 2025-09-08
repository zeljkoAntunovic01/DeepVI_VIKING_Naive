import jax
import jax.numpy as jnp
from jax import flatten_util

def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))

def estimate_sigma(params, apply_fn, y_train):
    y_pred = apply_fn(params)
    residuals = y_train - y_pred
    sigma2 = jnp.mean(jnp.square(residuals))
    return sigma2

def vectorize_nn(model_fn, params):
    """Vectorize the Neural Network
    Inputs:
    parameters: Pytree of parameters
    model_fn: A function that takes in pytree parameters and data

    Outputs:
    params_vec: Vectorized parameters
    unflatten_fn: Unflatten function
    model_apply_vec: A function that takes in vectorized parameters and data
    """
    params_vec, unflatten_fn = flatten_util.ravel_pytree(params)
    def model_apply_vec(params_vectorized, x):
        return model_fn(unflatten_fn(params_vectorized), x)
    return params_vec, unflatten_fn, model_apply_vec