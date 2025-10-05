import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sampling import compute_J, sample_theta

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

def plot_linearized_predictions(x_train, y_train, model_fn_vec, params_vec, thetas):
    """
    Plot linearized Bayesian model predictions:
    - Top: posterior samples (blue) + posterior mean (red) + training data (black)
    - Bottom: standard deviation (variance) across samples
    """

    x_test = jnp.linspace(-2, 1, 200).reshape(-1, 1)

    # Predict using linearized model for all theta samples
    def f_lin(theta):
        # Linearized: f(theta) ≈ f(theta_mean) + J(theta - theta_mean)
        f_map = model_fn_vec(params_vec, x_test).squeeze()  # baseline
        J = compute_J(params_vec, model_fn_vec, x_test)  # (N, D)
        return f_map + (theta - params_vec) @ J.T

    lin_preds = jax.vmap(f_lin)(thetas)  # (S, N)
    y_mean_lin = jnp.mean(lin_preds, axis=0)
    y_std_lin = jnp.std(lin_preds, axis=0)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top: posterior samples + mean + data
    for i in range(lin_preds.shape[0]):
        ax1.plot(x_test.squeeze(), lin_preds[i], color='skyblue', alpha=0.3, linewidth=1)
    ax1.plot(x_test.squeeze(), y_mean_lin, color='crimson', linewidth=2, label="Posterior mean")
    ax1.scatter(x_train, y_train, color='black', marker='x', s=40, label="Train data")

    ax1.set_title("Linearized Bayesian Model Predictions")
    ax1.legend(loc="upper right")
    ax1.set_ylabel("f(x)")

    # Bottom: standard deviation
    ax2.plot(x_test.squeeze(), y_std_lin, color='steelblue', linewidth=2)
    ax2.set_ylabel("Standard deviation")
    ax2.set_xlabel("x")

    plt.tight_layout()
    plt.savefig("results/plots/Linearized_Predictions.png", dpi=200)
    plt.close(fig)