import flax.linen as nn
import jax.numpy as jnp

class Linear_mnist(nn.Module):
    num_inputs: int
    num_labels: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.num_inputs)
        x = nn.Dense(self.num_labels)(x)
        return x
    
class MLP_fmnist(nn.Module):
    num_inputs: int
    num_labels: int
    units: list
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for unit in self.units:
            x = nn.Dense(unit)(x)
            x = getattr(nn, self.activation)(x)
        x = nn.Dense(self.num_labels)(x)
        return x
    
def get_model(config):
    if config.dataset_name == 'mnist':
        return Linear_mnist(num_inputs=config.num_inputs, num_labels=config.num_labels)
    elif config.dataset_name == 'fmnist':
        return MLP_fmnist(num_inputs=config.num_inputs, num_labels=config.num_labels, units=[64, 64], activation='relu')
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    