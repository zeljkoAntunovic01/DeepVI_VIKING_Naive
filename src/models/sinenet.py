# model.py
from flax import linen as nn
from jax import nn as jnn

class SineNet(nn.Module):
    out_dims: 1
    hidden_dim: 64
    num_layers: 3

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = jnn.tanh(x)
        x = nn.Dense(self.out_dims)(x)  # shape inference
        return x