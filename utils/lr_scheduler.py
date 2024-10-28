import jax.numpy as jnp

class LRScheduler:
  """Base class of simple scheduler, allowing to track current learning rate."""

  def __init__(self,
               learning_rate: jnp.float32,
               learning_rate_decay: jnp.float32,
               num_examples: jnp.int32,
               batch_size: jnp.int32,
               epochs: jnp.int32) -> None:
    """Constructs a learning rate scheduler.

    Args:
      learning_rate: base learning rate to start with
      learning_rate_decay: learning rate decay to be applied
      num_examples: number of examples per epoch
      batch_size: batch size used for training
      epochs: total number of epochs
    """
    self.base_learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.batch_size = batch_size
    self.num_examples = num_examples
    self.epochs = epochs

  def __call__(self, step: jnp.int32) -> float:
    """Applies learning rate schedule to compute current learning rate.

    Args:
      step: training step to compute learning rate for.

    Returns:
      Updated learning rate.
    """
    raise NotImplementedError


class ExponentialLRScheduler(LRScheduler):
  """Exponential learning rate schedule."""
  
  def __call__(self, step: jnp.int32) -> jnp.float32:
    steps_per_epoch = jnp.ceil(self.num_examples / self.batch_size)
    
    learning_rate_scaler = self.learning_rate_decay ** (step // steps_per_epoch)
    # self.current_learning_rate = learning_rate_scaler * self.base_learning_rate
    
    return self.current_learning_rate

class MultIStepLRScheduler(LRScheduler):
  """Multi-step learning rate schedule."""

  def __call__(self, step: jnp.int32) -> jnp.float32:
    steps_per_epoch = jnp.ceil(self.num_examples / self.batch_size)
    epoch = step // steps_per_epoch
    epochs_per_step = self.epochs // 5
    learning_rate_step = jnp.maximum(epoch//epochs_per_step - 1, 0)
    
    learning_rate_scaler = self.learning_rate_decay ** learning_rate_step
    self.current_learning_rate = learning_rate_scaler * self.base_learning_rate
    
    return self.current_learning_rate