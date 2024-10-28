import numpy as np
import jax
import jax.numpy as jnp
import optax

optimizer = optax.adam(1e-3)
params = jnp.array([1.0, 2.0, 3.0])
opt_state = optimizer.init(params)
grad = jnp.array([4.0, 5.0, 6.0])


@jax.jit
def foo(grad, opt_state):
    return optimizer.update(grad, opt_state)

print("Running...")
foo(grad, opt_state)
print("Success!")