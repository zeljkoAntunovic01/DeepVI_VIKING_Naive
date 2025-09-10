from typing import NamedTuple

import jax
import jax.numpy as jnp


# One example of what can be a "PyTree"
# (I'm doing this to avoid using any frameworks in this example)
class ModelParams(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array


def make_mlp(num_hidden):
    def init_fn(num_inputs=1, *, key):
        k_w1, k_b1, k_w2, k_b2, k_w3, k_b3 = jax.random.split(key, num=6)
        return ModelParams(
            jax.random.normal(k_w1, shape=(num_hidden, num_inputs)),
            jax.random.normal(k_b1, shape=(num_hidden,)),
            jax.random.normal(k_w2, shape=(num_hidden, num_hidden)),
            jax.random.normal(k_b2, shape=(num_hidden,)),
            jax.random.normal(k_w3, shape=(num_hidden,)),
            jax.random.normal(k_b3, shape=()),
        )

    def apply_fn(params: ModelParams, x_single):
        x = jnp.broadcast_to(x_single, (1,))
        x = params.w1 @ x + params.b1
        x = jnp.tanh(x)
        x = params.w2 @ x + params.b2
        x = jnp.tanh(x)
        x = params.w3 @ x + params.b3
        return x

    return init_fn, apply_fn


# This generates the data
def make_wave(key, num: int):
    std = jnp.linspace(1e-3, 1e0, num)
    x = jnp.linspace(0.35, 0.65, num)
    y = 5 * jnp.sin(10 * x)

    z = jax.random.normal(key, shape=y.shape)
    return x, y + std * z


# I tend to do the following to create data with the "gap" in the middle
key = jax.random.PRNGKey(seed=0)
key, key_data = jax.random.split(key)
x, y = make_wave(key_data, num=20)
x = jnp.concatenate((x[:5], x[-5:]))
y = jnp.concatenate((y[:5], y[-5:]))

# Model usage
key, key_model = jax.random.split(key)
model_init_fn, model_apply_fn = make_mlp(num_hidden=8)
model_params = model_init_fn(num_inputs=1, key=key_model)
vmapped_apply_fn = jax.vmap(model_apply_fn, in_axes=(None, 0))

# Predictions on entire data (note the `vmapped_apply_fn` usage, since
# `apply_fn` is defined on a single data point)
y_hat = vmapped_apply_fn(model_params, x)
