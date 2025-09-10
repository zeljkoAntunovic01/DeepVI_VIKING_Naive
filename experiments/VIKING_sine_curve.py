import jax
import jax.numpy as jnp
import optax
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import estimate_sigma, vectorize_nn
from src.data.sinedata import generate_data
from src.losses import sse_loss

N = 100
SEED = 42
NOISE_VAR = 0.01

# 1. Calculation of the projection matrix UUt derived from the GGN
def calculate_UUt(model_fn, params_vec, x_train, y_train):
    model_fn_lmbd = lambda p: jnp.squeeze(model_fn(p, x_train))
    J = jax.jacfwd(model_fn_lmbd)(params_vec)
    sigma2_est = estimate_sigma(params_vec, model_fn_lmbd, y_train)

    GGN = (1.0 / sigma2_est) * (J.T @ J)
    D, V = jnp.linalg.eigh(GGN)
    null_mask = jnp.where(D <= 1e-2, 1.0, 0.0)
    I_p = jnp.eye(GGN.shape[0])

    UUt = I_p - (V @ (1.0 - null_mask)) @ V.T
    return UUt

# 2. Function that samples theta from our posterior
def sample_theta(key, num_samples, UUt, theta_hat, sigma_ker, sigma_im):
    D = theta_hat.shape[0]
    subkeys = jax.random.split(key, num_samples)
    eps_samples = []
    eps_ker_samples = []
    
    def sample_fn(subkey):
        eps = jax.random.normal(subkey, (D,))
        eps_ker = UUt @ eps

        eps_samples.append(eps)
        eps_ker_samples.append(eps_ker)

        eps_im = eps - eps_ker
        return theta_hat + sigma_ker * eps_ker + sigma_im * eps_im
    
    thetas = jax.vmap(sample_fn)(subkeys)
    return thetas, eps_samples, eps_ker_samples

# 3. Reconstruction term of the ELBO
def reconstruction_term(model_fn_vec, thetas, x, y):
    B = x.shape[0]
    O = y.shape[-1]
    rho =  1 / NOISE_VAR
    log_rho = jnp.log(rho)

    def log_likelihood(theta):
        y_pred = model_fn_vec(theta, x)
        sse = jnp.sum(sse_loss(y_pred, y))
        log_prob = (
            -N * O / 2 * jnp.log(2 * jnp.pi)
            + N * O / 2 * log_rho
            - (N / B) * 0.5 * rho * sse
        )
        return log_prob

    log_likelihoods = jax.vmap(log_likelihood)(thetas)
    return jnp.mean(log_likelihoods)

# 4. KL term of the ELBO
def KL_term(prior_vec, sigma_ker, sigma_im, eps_samples, eps_ker_samples):
    pass

# 5. Loss function (negative ELBO) that needs to be optimized with gradient descent
def loss_fn(params_opt, model_fn_vec, UUt, x, y, sample_key):
    thetas, eps_samples, eps_ker_samples = sample_theta(
        key=sample_key,
        num_samples=100,
        UUt=UUt,
        theta_hat=params_opt["theta"],
        sigma_ker=params_opt["sigma_ker"],
        sigma_im=params_opt["sigma_im"]
    )

    rec_term = reconstruction_term(model_fn_vec, thetas, x, y)
    kl_term = KL_term()
    elbo = rec_term - kl_term
    
    return -elbo


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

    sigma_kernel_key, sigma_image_key = jax.random.split(jax.random.PRNGKey(SEED))
    sigma_kernel = jax.random.normal(sigma_kernel_key)
    sigma_image = jax.random.normal(sigma_image_key)
    _, sample_key = jax.random.split(jax.random.PRNGKey(SEED))

    params_opt = {
        "theta": params_vec,
        "sigma_ker": sigma_kernel,
        "sigma_im": sigma_image,
    }

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params_opt)

    # 6. Training step of our optimization algorithm for optimizing our Loss (-ELBO)
    @jax.jit
    def train_step(params_opt, opt_state, UUt, key):
        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(params_opt, model_fn_vec, UUt, x_train, y_train, subkey)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_opt = optax.apply_updates(params_opt, updates)
        return loss, params_opt, opt_state
    
    jit_train_step = jax.jit(train_step)
    num_epochs = 250  # or however many you want
    log_every = 25

    params_opt_current = params_opt
    opt_state_current = opt_state
    training_key = sample_key  # initial PRNG key

    for epoch in range(num_epochs):
        training_key, subkey = jax.random.split(training_key)

        UUt = calculate_UUt(model_fn_vec, params_opt_current["theta"], x_train, y_train)
        loss, params_opt_current, opt_state_current = jit_train_step(
            params_opt_current, opt_state_current, UUt, subkey
        )

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"[Epoch {epoch}] Loss (-ELBO): {loss:.4f}")


if __name__ == "__main__":
    main()
