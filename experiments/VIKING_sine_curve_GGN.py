import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sinenet import SineNet
from src.sampling import calculate_UUt, calculate_UUt_svd, sample_theta
from src.utils import vectorize_nn
from src.losses import sse_loss
from src.plotting.naive_sine_plots import plot_bayesian_samples_with_mean, plot_mean_bayesian_with_MAP, predict_and_plot_bayesian_mean_for_epoch
from src.data.sinedata import generate_data, generate_sine_data

N = 10
SEED = 42
NOISE_VAR = 0.96

# Reconstruction term of the ELBO
def reconstruction_term(model_fn_vec, thetas, x, y):
    rho = 1 / NOISE_VAR

    def log_likelihood(theta):
        y_pred = model_fn_vec(theta, x)
        log_prob = -0.5 * rho * sse_loss(y_pred, y)
        return log_prob

    log_likelihoods = jax.vmap(log_likelihood)(thetas)
    return jnp.mean(log_likelihoods)

# KL term of the ELBO
def KL_term(prior_vec, theta_hat, sigma_ker, sigma_im, eps_samples, eps_ker_samples, rank_ker=None):
    sigma_ker_2 = jnp.exp(2*sigma_ker)
    sigma_im_2 = jnp.exp(2*sigma_im)
    prior_vec_inv = 1.0 / prior_vec
    num_samples, D = eps_samples.shape
    hadamard_eps = eps_samples * eps_ker_samples

    trace_products = hadamard_eps @ prior_vec_inv
    trace = ((sigma_ker_2 - sigma_im_2) / num_samples) * jnp.sum(trace_products) + sigma_im_2 * jnp.sum(prior_vec_inv)

    hadamard_theta = jnp.square(theta_hat)
    middle_term = jnp.dot(prior_vec_inv, hadamard_theta)

    ln_det_prior = jnp.sum(jnp.log(prior_vec))

    R = jnp.mean(jnp.sum(eps_samples * eps_ker_samples, axis=1))
    ln_det_post = 2 * R * sigma_ker + 2 * (D - R) * sigma_im

    return 0.5 * (trace - D + middle_term + ln_det_prior - ln_det_post)

# KL Term from the paper for debugging purposes
def KL_term_alpha(theta_hat, sigma_ker, sigma_im, eps_samples, eps_ker_samples, rank_ker=None):
    sigma_ker_2 = jnp.exp(2*sigma_ker)
    sigma_im_2 = jnp.exp(2*sigma_im)
    alpha = 1.0 / sigma_ker_2
    num_samples, D = eps_samples.shape

    if rank_ker is not None:
        R = rank_ker
    else:
        # Fallback if not passed
        R = jnp.mean(jnp.sum(eps_samples * eps_ker_samples, axis=1))

    # Calculate Tr(Σ)
    trace = sigma_ker_2 * R + sigma_im_2 * (D - R)

    # Ln(Σ)
    ln_det_post = 2 * R * sigma_ker + 2 * (D - R) * sigma_im

    return 0.5 * (
        alpha * trace - D + alpha * (jnp.pow(jnp.linalg.norm(theta_hat), 2)) - D * jnp.log(alpha) - ln_det_post
    )

# 5. Loss function (negative ELBO) that needs to be optimized with gradient descent
def loss_fn(params_opt, model_fn_vec, UUt, x, y, sample_key, prior_vec, rank_ker=None):
    thetas, eps_samples, eps_ker_samples = sample_theta(
        key=sample_key,
        num_samples=100,
        UUt=UUt,
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"]
    )

    # Reconstruction part
    rec_term = reconstruction_term(model_fn_vec, thetas, x, y)

    # KL term
    """ kl = KL_term(
        prior_vec=prior_vec,
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"],
        eps_samples=eps_samples,
        eps_ker_samples=eps_ker_samples,
        rank_ker=rank_ker
    ) """

    kl = KL_term_alpha(
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"],
        eps_samples=eps_samples,
        eps_ker_samples=eps_ker_samples,
        rank_ker = rank_ker
    )

    elbo = rec_term - kl
    
    return -elbo, (rec_term, kl)

def main():
    prior_key, post_key, model_key, data_key = jax.random.split(jax.random.PRNGKey(SEED), num=4)

    with open("./checkpoints/sine_regression.pickle", "rb") as f:
        checkpoint = pickle.load(f)

    #x_train = checkpoint["train_stats"]["x_train"]
    #y_train = checkpoint["train_stats"]["y_train"]
    x_train, y_train = generate_data(n_train=10, key=data_key)

    params_dict_map = checkpoint["params"]
    model_map = checkpoint["train_stats"]["model"]

    model_new = SineNet(out_dims=1, hidden_dim=8 ,num_layers=2)
    params_new = model_new.init(model_key, x_train[:2])

    params_vec, unflatten, model_fn_vec = vectorize_nn(lambda p, x: model_new.apply(p, x), params_new)
    params_map_vec, unflatten, model_map_fn_vec = vectorize_nn(model_map.apply, params_dict_map)

    x_test = jnp.linspace(-2, 1, 200).reshape(-1, 1)
    y_map = model_map_fn_vec(params_map_vec, x_test).squeeze()

    prior_vec = jnp.clip(jax.random.normal(prior_key, (params_vec.shape[0],)) ** 2, 0.1, 10.0)

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

    schedule = optax.exponential_decay(1e-2, 100, 0.99, 1e-5)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params_opt)

    @jax.jit
    def train_step(params_opt, opt_state, UUt, key, rank_ker):
        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (rec_term, kl)), grads = grad_fn(params_opt, model_fn_vec, UUt, x_train, y_train, subkey, prior_vec, rank_ker)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return loss, params_opt, opt_state, rec_term, kl

    num_epochs, log_every, plot_epochs = 3000, 75, [5, 25, 50, 100, 250, 500, 1000, 2000]
    params_opt_current, opt_state_current, training_key = params_opt, opt_state, sample_key

    for epoch in range(num_epochs):
        training_key, subkey = jax.random.split(training_key)
        UUt, rank_ker = calculate_UUt_svd(model_fn_vec, params_opt_current["theta"], x_train, y_train)
        loss, params_opt_current, opt_state_current, rec_term, kl = train_step(params_opt_current, opt_state_current, UUt, subkey, rank_ker)

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"[Epoch {epoch}] Loss (-ELBO): {loss:.4f}")
            print(f"Rec term = {rec_term:.4f} ||| KL Term = {kl:.4f}")
            print(f"Sigma kernel: {jnp.exp(params_opt_current['sigma_ker']):.4f}, Sigma image: {jnp.exp(params_opt_current['sigma_im']):.4f}, Alpha: {(1.0/ (jnp.exp(params_opt_current['sigma_ker']) ** 2)):.2f}")
        
        if epoch in plot_epochs:
            predict_and_plot_bayesian_mean_for_epoch(post_key, model_fn_vec, params_opt_current, UUt, y_map, x_train, y_train, epoch)

    thetas, _, _ = sample_theta(post_key, 50, UUt, params_opt_current["theta"], params_opt_current["sigma_ker"], params_opt_current["sigma_im"])
    y_preds = jax.vmap(lambda t: model_fn_vec(t, x_test))(thetas)
    y_mean, y_std = jnp.mean(y_preds, axis=0).squeeze(), jnp.std(y_preds, axis=0).squeeze()

    plot_mean_bayesian_with_MAP(x_train, y_train, x_test, y_mean, y_map, y_std)
    plot_bayesian_samples_with_mean(x_train, y_train, x_test, y_mean, y_preds)

    with open(f"./checkpoints/sine_VIKING_GGN.pickle", "wb") as file:
        pickle.dump({"elbo_params": {"sigma_ker": params_opt_current["sigma_ker"], "sigma_im": params_opt_current["sigma_im"], "theta": params_opt_current["theta"]}}, file)

if __name__ == "__main__":
    main()
