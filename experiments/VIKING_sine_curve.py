import jax
import jax.numpy as jnp
import tree_math as tm
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import estimate_sigma, vectorize_nn
from src.data.sinedata import generate_data

SEED = 42

def calculate_UUt(model_fn, params_vec, x_train, y_train):
    model_fn_lmbd = lambda p: jnp.squeeze(model_fn(p, x_train))
    J = jax.jacfwd(model_fn_lmbd)(params_vec)
    sigma2_est = estimate_sigma(params_vec, model_fn_lmbd, y_train)

    GGN = (1.0 / sigma2_est) * (J.T @ J)
    D, V = jnp.linalg.eigh(GGN)
    null_mask = (D <= 1e-2)
    I_p = jnp.eye(GGN.shape[0])

    UUt = I_p - (V @ (1.0, - null_mask)) @ V.T
    return UUt

def main():
    _, data_key = jax.random.split(jax.random.PRNGKey(SEED))
    x_train, y_train = generate_data(key=data_key)

    # Load model, extract params...
    with open("./checkpoints/sine_regression.pickle", "rb") as f:
        checkpoint = pickle.load(f)

    params_dict = checkpoint["params"]
    params = params_dict['params']
    model = checkpoint["train_stats"]["model"]
    params_vec, unflatten, model_fn_vec = vectorize_nn(model.apply, params_dict)

    UUt = calculate_UUt(model_fn_vec, params_vec, x_train, y_train)
    

if __name__ == "__main__":
    main()
