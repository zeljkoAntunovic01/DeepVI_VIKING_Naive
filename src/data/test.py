import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

NOISE_VAR = 0.01
N = 50
SEED = 42
def generate_data(n_train=100, noise_var = 0.01, key=jax.random.PRNGKey(0)):
    key_1, key_2, key_3 = jax.random.split(key, 3)
    X_1 = jax.random.uniform(key_1, shape=(n_train//2, 1), minval=-1.5, maxval=-1.0)
    X_2 = jax.random.uniform(key_2, shape=(n_train//2, 1), minval=-0.5, maxval=0)
    X = jnp.concatenate([X_1, X_2], axis=0)
    cosx = jnp.cos(4 * X + 0.8)
    randn = jax.random.normal(key_3, shape=(n_train, 1))*jnp.sqrt(noise_var)
    Y = cosx + randn
    return X, Y

model_key, data_key = jax.random.split(jax.random.PRNGKey(SEED))
x_train, y_train = generate_data(n_train=N, noise_var=NOISE_VAR, key=data_key)
plt.plot(x_train, y_train, 'o')
plt.show()