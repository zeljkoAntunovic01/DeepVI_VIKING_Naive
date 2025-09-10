import matplotlib.pyplot as plt


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