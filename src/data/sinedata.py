import jax
import jax.numpy as jnp

def f(x):
    return jnp.cos(4 * x + 0.8)

def generate_data(n_train=100, noise_var = 0.01, key=jax.random.PRNGKey(0)):
    key_1, key_2, key_3 = jax.random.split(key, 3)
    X_1 = jax.random.uniform(key_1, shape=(n_train//2, 1), minval=-1.5, maxval=-1.0)
    X_2 = jax.random.uniform(key_2, shape=(n_train//2, 1), minval=-0.5, maxval=0)
    X = jnp.concatenate([X_1, X_2], axis=0)
    cosx = jnp.cos(4 * X + 0.8)
    randn = jax.random.normal(key_3, shape=(n_train, 1))*jnp.sqrt(noise_var)
    Y = cosx + randn
    return X, Y

def generate_sine_data(n_train, key):
    std = jnp.linspace(1e-3, 1e0, n_train)
    x = jnp.linspace(0.35, 0.65, n_train)
    #y = 5 * jnp.sin(10 * x)

    y = 0.1 * jnp.sin(10 * x)
    z = jax.random.normal(key, shape=y.shape)
    return x, y + std * z
