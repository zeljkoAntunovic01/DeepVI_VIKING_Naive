import jax
import jax.numpy as jnp
import optax
import os
import matplotlib.pyplot as plt
import tree_math as tm
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sinenet import SineNet
from src.data.sinedata import generate_data, f
from src.utils import compute_num_params
from src.losses import sse_loss

N = 100
BATCH_SIZE = 10
NOISE_VAR = 0.01
SEED = 42
ALPHA = 1.0
N_EPOCH = 1000

def main():
    model_key, data_key = jax.random.split(jax.random.PRNGKey(SEED))
    x_train, y_train = generate_data(n_train=N, noise_var=NOISE_VAR, key=data_key)
    x_val = jnp.linspace(-2, 2, 100).reshape(-1, 1)
    y_val = f(x_val)

    model = SineNet(out_dims=1, hidden_dim=10, num_layers=2)
    params = model.init(model_key, x_train[:BATCH_SIZE])
    n_batches = N // BATCH_SIZE
    rho =  1 / NOISE_VAR
    D = compute_num_params(params)

    log_alpha = jnp.log(ALPHA)
    log_rho = jnp.log(rho)

    lr = 1e-3
    optim = optax.adam(lr)
    opt_state = optim.init(params)

    def map_loss(params, x, y):
        B = x.shape[0]
        O = y.shape[-1]
        out = model.apply(params, x)
        vparams = tm.Vector(params)
        log_likelihood = (
            -N * O / 2 * jnp.log(2 * jnp.pi)
            + N * O / 2 * log_rho
            - (N / B) * 0.5 * rho * jnp.sum(jax.vmap(sse_loss)(out, y))
        )
        log_prior = -D / 2 * jnp.log(2 * jnp.pi) + D / 2 * log_alpha - 0.5 * ALPHA * vparams @ vparams
        loss = log_likelihood + log_prior
        return -loss, (log_likelihood, log_prior)
    
    def train_step(params, opt_state, x, y):
        grad_fn = jax.value_and_grad(map_loss, argnums=0, has_aux=True)
        loss, grads = grad_fn(params, x, y)
        param_updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, param_updates)
        return loss, params, opt_state

    jit_train_step = jax.jit(train_step)
    losses = []
    log_likelihoods = []
    log_priors = []

    for epoch in range(1, N_EPOCH + 1):
        shuffle_key, split_key = jax.random.split(data_key)
        batch_indices_shuffled = jax.random.permutation(shuffle_key, x_train.shape[0])

        for i in range(n_batches):
            x_batch = x_train[batch_indices_shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]]
            y_batch = y_train[batch_indices_shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]]

            loss, params, opt_state = jit_train_step(
                params, opt_state, x_batch, y_batch
            )

            loss, (log_likelihood, log_prior) = loss
            losses.append(loss)
            log_likelihoods.append(log_likelihood.item())
            log_priors.append(log_prior.item())

        log_likelihood_epoch_loss = jnp.mean(jnp.array(log_likelihoods[-BATCH_SIZE:]))
        epoch_loss = jnp.mean(jnp.array(losses[-BATCH_SIZE:]))
        epoch_prior = jnp.mean(jnp.array(log_priors[-BATCH_SIZE:]))
        
        if (epoch % 25 == 0):
            print(f"Epoch {epoch} | log likelihood loss={log_likelihood_epoch_loss:.2f} | loss ={epoch_loss:.2f} | prior loss={epoch_prior:.2f}")


    plt.plot(x_train, y_train, 'o')
    plt.plot(x_val, y_val, label="True function")
    plt.plot(x_val, model.apply(params, x_val), label="Predictions")
    plt.legend()
    plt.title("SineNet Fit")
    plt.show()

    # Save Learned parameters
    train_stats_dict = {}
    train_stats_dict['x_train'] = x_train
    train_stats_dict['y_train'] = y_train
    train_stats_dict['x_val'] = x_val
    train_stats_dict['y_val'] = y_val
    train_stats_dict['model'] = model
    train_stats_dict['n_params'] = D

    with open(f"./checkpoints/sine_regression.pickle", "wb") as file:
        pickle.dump(
            {"params": params, "alpha": ALPHA, "rho": rho, "train_stats": train_stats_dict}, file
        )


if __name__ == "__main__":
    main()
