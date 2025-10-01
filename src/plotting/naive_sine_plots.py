import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sampling import sample_theta
from src.utils import vectorize_nn

def plot_mean_bayesian_with_MAP(x_train, y_train, x_test, y_mean, y_map, y_std):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, color="black", alpha=0.6, label="Train data")
    plt.plot(x_test, y_mean, label="Posterior mean", linewidth=2)
    plt.plot(x_test, y_map, color="red", linestyle="--", linewidth=2, label="MAP estimate")

    # 2σ credible interval
    plt.fill_between(
        x_test.squeeze(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        alpha=0.3,
        label="Uncertainty (±2σ)"
    )

    plt.legend()
    plt.title("Bayesian Sine Regression (VIKING Naive VI)")
    plt.savefig("results/plots/Mean_Bayesian_with_MAP.png")


def plot_bayesian_samples_with_mean(x_train, y_train, x_test, y_mean, y_preds):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, color="black", alpha=0.6, label="Train data")

    # Posterior samples (thin blue lines)
    for i in range(min(30, y_preds.shape[0])):  # limit to 30 to avoid clutter
        plt.plot(x_test, y_preds[i], color="blue", alpha=0.3, linewidth=1)

    # Posterior mean (thick red line)
    plt.plot(x_test, y_mean, color="red", linewidth=2.5, label="Posterior mean")

    plt.legend()
    plt.title("Bayesian Sine Regression (VIKING Naive VI)")
    plt.savefig("results/plots/VIKING_Bayesian_Samples.png")


def predict_and_plot_bayesian_mean_for_epoch(post_key, model_fn_vec, params_opt_current, UUt, y_map, x_train, y_train, epoch):
    x_test = jnp.linspace(-2, 1, 200).reshape(-1, 1)
    thetas, _, _ = sample_theta(post_key, 50, UUt, params_opt_current["theta"], params_opt_current["sigma_ker"], params_opt_current["sigma_im"])
    y_preds = jax.vmap(lambda t: model_fn_vec(t, x_test))(thetas)
    y_mean, y_std = jnp.mean(y_preds, axis=0).squeeze(), jnp.std(y_preds, axis=0).squeeze()

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, color="black", alpha=0.6, label="Train data")
    plt.plot(x_test, y_mean, label="Posterior mean", linewidth=2)
    plt.plot(x_test, y_map, color="red", linestyle="--", linewidth=2, label="MAP estimate")

    # 2σ credible interval
    plt.fill_between(
        x_test.squeeze(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        alpha=0.3,
        label="Uncertainty (±2σ)"
    )

    plt.legend()
    plt.title(f"Bayesian Sine Regression (VIKING Naive VI) - Epoch {epoch}")
    plt.savefig(f"results/plots/per_epoch/prior_vec/Mean_Bayesian_with_MAP_epoch_{epoch}.png")
