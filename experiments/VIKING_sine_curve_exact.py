import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sampling import compute_J, sample_theta_exact
from src.utils import vectorize_nn
from src.losses import sse_loss
from src.plotting.naive_sine_plots import plot_bayesian_samples_with_mean, plot_mean_bayesian_with_MAP

N = 150
SEED = 42
NOISE_VAR = 0.96

# Reconstruction term of the ELBO
def reconstruction_term(model_fn_vec, thetas, x, y):
    B = x.shape[0]
    O = y.shape[-1]
    rho =  1 / NOISE_VAR
    log_rho = jnp.log(rho)

    def log_likelihood(theta):
        y_pred = model_fn_vec(theta, x)
        sse = sse_loss(y_pred, y)
        log_prob = (
            -N * O / 2 * jnp.log(2 * jnp.pi)
            + N * O / 2 * log_rho
            - (N / B) * 0.5 * rho * sse
        )
        return log_prob

    log_likelihoods = jax.vmap(log_likelihood)(thetas)
    return jnp.mean(log_likelihoods)

# KL term of the ELBO
def KL_term(prior_vec, theta_hat, sigma_ker, sigma_im, eps_samples, eps_ker_samples):
    sigma_ker_2 = jnp.exp(sigma_ker) ** 2
    sigma_im_2 = jnp.exp(sigma_im) ** 2
    prior_vec_inv = 1.0 / prior_vec
    num_samples, D = eps_samples.shape
    hadamard_eps = eps_samples * eps_ker_samples # (num_samples, D)

    # Calculate Tr(Σ_p^{-1} Σ)
    trace_products = hadamard_eps @ prior_vec_inv # (num_samples, 1)
    trace = ((sigma_ker_2 - sigma_im_2) / num_samples) * jnp.sum(trace_products) + sigma_im_2 * jnp.sum(prior_vec_inv)

    # Quadratic term (theta^T Σ_p^{-1} theta)
    hadamard_theta = jnp.square(theta_hat)
    middle_term = jnp.dot(prior_vec_inv, hadamard_theta)

    ln_det_prior = jnp.sum(jnp.log(prior_vec))

    R = jnp.mean(jnp.sum(eps_samples * eps_ker_samples, axis=1))  # shape (S,) -> mean -> shape(1,)
    ln_det_post = 2 * R * sigma_ker + 2 * (D - R) * sigma_im

    return 0.5 * (
        trace - D + middle_term + ln_det_prior - ln_det_post
    )

# KL Term from the paper for debugging purposes
def KL_term_alpha(theta_hat, sigma_ker, sigma_im, eps_samples, eps_ker_samples):
    sigma_ker_2 = jnp.exp(sigma_ker) ** 2
    sigma_im_2 = jnp.exp(sigma_im) ** 2
    alpha = 1.0 / sigma_ker_2
    num_samples, D = eps_samples.shape

    R = jnp.mean(jnp.sum(eps_samples * eps_ker_samples, axis=1))  # shape (S,) -> mean -> shape(1,)

    # Calculate Tr(Σ)
    trace = sigma_ker_2 * R + sigma_im_2 * (D - R)

    # Ln(Σ)
    ln_det_post = 2 * R * sigma_ker + 2 * (D - R) * sigma_im

    return 0.5 * (
        alpha * trace - D + alpha * (jnp.linalg.norm(theta_hat) ** 2) - D * jnp.log(alpha) - ln_det_post
    )


def loss_fn(params_opt, model_fn_vec, x, y, sample_key):
    # Step 1: Build J at current params
    J = compute_J(params_opt["theta"], model_fn_vec, x, y)

    # Step 2: Sample using exact projection
    thetas, eps_samples, eps_ker_samples = sample_theta_exact(
        key=sample_key,
        num_samples=100,
        J=J,
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"]
    )

    # Step 3: Reconstruction
    rec_term = reconstruction_term(model_fn_vec, thetas, x, y)

    # Step 4: KL
    """ kl = KL_term(
        prior_vec=prior_vec,
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"],
        eps_samples=eps_samples,
        eps_ker_samples=eps_ker_samples
    ) """

    kl = KL_term_alpha(
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"],
        eps_samples=eps_samples,
        eps_ker_samples=eps_ker_samples
    )

    elbo = rec_term - kl
    return -elbo, (rec_term, kl)

def main():
    prior_key, post_key = jax.random.split(jax.random.PRNGKey(SEED), num=2)

    # Load model, extract params...
    with open("./checkpoints/sine_regression.pickle", "rb") as f:
        checkpoint = pickle.load(f)

    x_train = checkpoint["train_stats"]["x_train"]
    y_train = checkpoint["train_stats"]["y_train"]

    if (x_train.shape[0] != N):
        raise (f"Train data length does not match the value of N={N}")
    params_dict = checkpoint["params"]
    model = checkpoint["train_stats"]["model"]
    params_vec, unflatten, model_fn_vec = vectorize_nn(model.apply, params_dict)
    prior_vec = jnp.clip(jax.random.normal(prior_key, (params_vec.shape[0],)) ** 2, 0.1, 10.0) # Vector of prior covariance diagonal values, sigmas squared

    sigma_kernel = jnp.exp(0.0)
    sigma_image = jnp.exp(-2.0)
    print(f"Initial log sigma_kernel: {jnp.log(sigma_kernel)}, Initial log sigma_image: {jnp.log(sigma_image)}")
    _, sample_key = jax.random.split(jax.random.PRNGKey(SEED))

    params_opt = {
        #"prior_vec": prior_vec,
        "theta": params_vec,
        "sigma_ker": jnp.log(sigma_kernel),
        "sigma_im": jnp.log(sigma_image),
    }

    schedule = optax.exponential_decay(
        init_value=1e-2,   # starting LR
        transition_steps=100,  # how often to decay (in steps)
        decay_rate=0.99,   # multiply LR by this every transition
        end_value=1e-5     # minimum LR
    )

    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params_opt)

    num_epochs = 3000  # or however many you want
    log_every = 75

    # Training step of our optimization algorithm for optimizing our Loss (-ELBO)
    @jax.jit
    def train_step(params_opt, opt_state, key):
        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (rec_term, kl)), grads = grad_fn(
            params_opt, model_fn_vec, x_train, y_train, subkey, #prior_vec
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return loss, params_opt, opt_state, rec_term, kl
    
    jit_train_step = jax.jit(train_step)

    params_opt_current = params_opt
    opt_state_current = opt_state
    training_key = sample_key  # initial PRNG key

    for epoch in range(num_epochs):
        training_key, subkey = jax.random.split(training_key)
        loss, params_opt_current, opt_state_current, rec_term, kl = jit_train_step(
            params_opt_current, opt_state_current, subkey
        )

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"[Epoch {epoch}] Loss (-ELBO): {loss:.4f}")
            print(f"Rec term = {rec_term:.4f} ||| KL Term = {kl:.4f}")


    x_test = jnp.linspace(-3, 3, 200).reshape(-1, 1)
    J = compute_J(params_opt_current["theta"], model_fn_vec, x_train, y_train)
    thetas, _, _ = sample_theta_exact(
        key=post_key,
        num_samples=100,  # number of posterior samples to draw
        J=J,
        theta_hat=params_opt_current["theta"],
        sigma_ker=params_opt_current["sigma_ker"],
        sigma_im=params_opt_current["sigma_im"]
    )

    # Forward pass for all posterior samples
    def predict(theta):
        return model_fn_vec(theta, x_test)

    y_preds = jax.vmap(predict)(thetas)

    # Mean prediction and uncertainty intervals
    y_mean = jnp.mean(y_preds, axis=0).squeeze()
    y_std = jnp.std(y_preds, axis=0).squeeze()

    # MAP prediction (red line)
    y_map = model_fn_vec(params_vec, x_test).squeeze()

    plot_mean_bayesian_with_MAP(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_mean=y_mean,
        y_map=y_map,
        y_std=y_std
    )

    plot_bayesian_samples_with_mean(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_mean=y_mean,
        y_preds=y_preds
    )

    elbo_params_dict = {}
    elbo_params_dict["sigma_ker"] = params_opt_current["sigma_ker"]
    elbo_params_dict["sigma_im"] = params_opt_current["sigma_im"]
    elbo_params_dict["theta"] = params_opt_current["theta"]

    with open(f"./checkpoints/sine_VIKING_ALPHA_exact.pickle", "wb") as file:
        pickle.dump(
            {"elbo_params": elbo_params_dict}, file
        )

if __name__ == "__main__":
    main()
