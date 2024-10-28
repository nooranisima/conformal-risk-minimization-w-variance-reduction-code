
from data.data_loader import load_data #this is new 
from models.models import get_model #this is new
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import pickle
import jax
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt
from tqdm import tqdm 
import flax.linen as nn
import optax
from operator import getitem

# from lr_scheduler import MultIStepLRScheduler
from utils.lr_scheduler import MultIStepLRScheduler
from utils.smooth_quantile import smooth_quantile

# Evaluations
from evaluation.evaluation import evaluate_conformal_prediction, compute_accuracy
# I like a colorful terminal :)
from rich import print
from rich.progress import track
from rich.live import Live
from rich.table import Table
from rich.progress import Progress
from rich.traceback import install
install()

key = random.PRNGKey(0)

'''def base_loss(params, x, y):
    logits = model.apply(params, x)
    probabilities = nn.softmax(logits)
    y_encoded = jax.nn.one_hot(y, num_classes)
    return -jnp.sum(y_encoded * jax.nn.log_softmax(probabilities)) / len(x)

def regularizer(params):
    return sum(jnp.sum(jnp.square(param)) for param in jax.tree.leaves(params)) # L2 regularization'''

def base_loss(params, x, y):
    logits = model.apply(params, x)
    logits = nn.log_softmax(logits)
    loss = jax.vmap(getitem)(logits, y)
    loss = -loss.mean()
    return loss



def main(config_func, num_trials = 1):
    global model, temperature, target_size, alpha, regularizer_weight, base_loss_weight, coverage_weight, size_weight, loss_matrix, epochs, train_loader, num_classes, num_inputs
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_accuracies = []
    epoch_variances = []
    epoch_set_sizes = []
    trial_params_and_seeds = []

    for trial in range(num_trials):
        config = config_func()

        epochs = config.epochs
        
        # Initialize dictionaries for storing metrics
        trial_train_losses = []
        trial_test_losses = []
        trial_test_accuracies = []
        trial_set_sizes = []

        #Set Random Seeds
        seed = torch.randint(0,10000, (1,)).item()
        torch.manual_seed(seed)
        np.random.seed(seed)
        key = random.PRNGKey(seed)

        # Set parameters, model, data based on configuration function
        batch_input_shape = config.batch_input_shape
        num_inputs = config.num_inputs
        num_classes = config.num_labels
        temperature = config.temperature
        target_size = config.target_size
        alpha = config.alpha
        regularizer_weight = config.regularizer_weight
        base_loss_weight = config.base_loss_weight
        coverage_weight = config.coverage_weight
        size_weight = config.size_weight
        loss_matrix = config.loss_matrix

        #Set the Model and Dataloaders
        model = get_model(config)
        train_loader, calib_loader, test_loader = load_data(config, seed = seed)

        #Initialize Optimizer, Scheduler, and Params
        epochs = 50
        lr_scheduler = MultIStepLRScheduler(
            learning_rate=0.05,
            learning_rate_decay=0.1,
            num_examples=len(train_loader.dataset),
            batch_size=config.batch_size,
            epochs=epochs
        )
        '''optimizer = optax.chain(
            optax.trace(decay=0, nesterov=True),
            optax.scale_by_schedule(lambda step: -lr_scheduler(step))
            )'''
        
        optimizer = optax.adam(learning_rate=0.05)
        
        dummy_batch_input = jnp.ones(batch_input_shape)
        params = model.init(key, dummy_batch_input)
        opt_state = optimizer.init(params)

        # Storing training loss
        trial_train_losses = []
        trial_test_losses = []
        trial_test_accuracies = []

        #Beginning of Epoch
        for epoch in range(epochs):
            running_loss = 0.0
            batch_losses = []

            for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = jnp.array(x), jnp.array(y)

                loss, grad = jax.value_and_grad(base_loss)(params, x, y)
                batch_losses.append(loss)
                running_loss += loss

                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)


                #End of Epoch
            
            # Statistics Calculations at the end of Epoch
            epoch_train_loss = running_loss/len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_train_loss:.4f}")

            # Evaluate Loss on the Test Set
            epoch_test_loss = jnp.mean(
                jnp.array([
                    base_loss(params, jnp.array(x_test), jnp.array(y_test))
                    for x_test, y_test in test_loader
                ]))
            print(f"Epoch [{epoch+1}/{epochs}] Test Loss: {epoch_test_loss:.4f}")

            # Evaluate Accuracy on the test Set
            test_accuracy = compute_accuracy(model, params, test_loader)
            print("Accuracy:", test_accuracy)
            # Evaluate Conformal Prediction
            _, _, avg_set_size, _ = evaluate_conformal_prediction(model, params, calib_loader, test_loader, num_trials=1, alpha=0.01)

            trial_train_losses.append(epoch_train_loss)
            trial_test_losses.append(epoch_test_loss)
            trial_test_accuracies.append(test_accuracy)
            trial_set_sizes.append(avg_set_size)

        # Store the parameters and seeds for this trial
        trial_params_and_seeds.append((params,seed))
        epoch_train_losses.append(trial_train_losses)
        epoch_test_losses.append(trial_test_losses)
        epoch_test_accuracies.append(trial_test_accuracies)
        epoch_set_sizes.append(trial_set_sizes)

    # Compute average metrics across trials for each epoch
    avg_epoch_train_losses = np.mean(np.array(epoch_train_losses), axis=0)
    std_epoch_train_losses = np.std(np.array(epoch_train_losses), axis = 0)

    avg_epoch_test_losses = np.mean(np.array(epoch_test_losses), axis=0)
    std_epoch_test_losses = np.std(np.array(epoch_test_losses), axis=0)

    avg_epoch_test_accuracies = np.mean(np.array(epoch_test_accuracies), axis=0)
    std_epoch_test_accuracies = np.std(np.array(epoch_test_accuracies), axis=0)
    avg_epoch_variances = np.mean(epoch_variances)

    avg_epoch_set_sizes = np.mean(np.array(epoch_set_sizes), axis = 0 )
    std_epoch_set_sizes = np.std(np.array(epoch_set_sizes), axis = 0 )
    print("set size ", avg_epoch_set_sizes)
    return (trial_params_and_seeds,
            (avg_epoch_train_losses,  std_epoch_train_losses),
            (avg_epoch_test_losses, std_epoch_test_losses),
            (avg_epoch_test_accuracies, std_epoch_test_accuracies),
            avg_epoch_variances,
            (avg_epoch_set_sizes, std_epoch_set_sizes))






            

            








