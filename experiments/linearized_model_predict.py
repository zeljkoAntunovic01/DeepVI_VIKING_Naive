import pickle
import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sinenet import SineNet
from src.plotting.naive_sine_plots import plot_linearized_predictions
from src.sampling import calculate_UUt_svd, compute_J, sample_theta
from src.utils import vectorize_nn

SEED = 42

def linearized_sample_predictions(model_fn_vec, theta_mean, thetas, x_train, y_train):
    """
    Predicts using linearized model f_lin(θ) = f(θ_mean) + J(θ - θ_mean)
    """
    f_map = model_fn_vec(theta_mean, x_train).squeeze()  # (N,)
    J = compute_J(theta_mean, model_fn_vec, x_train, y_train)  # (N, D)
    deltas = thetas - theta_mean  # (S, D)
    preds = deltas @ J.T + f_map  # (S, N)
    return preds

if __name__ == "__main__":
    post_key, next_key = jax.random.split(jax.random.PRNGKey(SEED), num=2)

    # Load saved stuff
    with open("./checkpoints/sine_VIKING_GGN.pickle", "rb") as f:
        elbo_checkpoint = pickle.load(f)
    
    elbo_stats = elbo_checkpoint["elbo_params"]
    
    model = SineNet(**elbo_stats["model_config"])
    params_dict = elbo_stats["theta"]
    log_sigma_ker = elbo_stats["sigma_ker"]
    log_sigma_im = elbo_stats["sigma_im"]

    params_vec, _, model_fn_vec = vectorize_nn(model.apply, params_dict)

    with open("./checkpoints/sine_regression.pickle", "rb") as f:
        train_checkpoint = pickle.load(f)

    x_train = train_checkpoint["train_stats"]["x_train"]
    y_train = train_checkpoint["train_stats"]["y_train"]


    # Sample from posterior
    UUt, _ = calculate_UUt_svd(model_fn_vec, params_vec, x_train, y_train)
    thetas, _, _ = sample_theta(post_key, 50, UUt, params_vec, log_sigma_ker, log_sigma_im)

    plot_linearized_predictions(x_train, y_train, model_fn_vec, params_vec, thetas)