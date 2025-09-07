import jax.numpy as jnp

def sse_loss(preds, y):
    residual = preds - y
    return jnp.sum(residual**2)