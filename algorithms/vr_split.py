
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

def tree_zero(params):
    return jax.tree_util.tree_map(lambda p: 0*jnp.asarray(p).astype(jnp.asarray(p).dtype), params)

def tree_sum(params, delta_params): # TODO: modify to allow for weighted sums
    return jax.tree_util.tree_map(
      lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype),
      params, delta_params)

def tree_scale(params, scale):
    return jax.tree_util.tree_map(lambda p: scale*jnp.asarray(p).astype(jnp.asarray(p).dtype), params)


@jit
def conformity_score_all_labels(params, x):
    logits,_ = model.apply(params, x, mutable= ["batch_stats"])
    # return nn.softmax(logits)          # THR
    # return logits                      # THR-L
    return jax.nn.log_softmax(logits) # THR-LP

@jit
def conformity_score(params, x, y):
    scores = conformity_score_all_labels(params, x)
    score = jnp.take_along_axis(scores, y[:, None], axis=1).flatten()
    return score

@jit
def smooth_predict_set(params, x, tau):
    scores = conformity_score_all_labels(params, x)
    return nn.sigmoid( (scores - tau) / temperature)

@jit
def smooth_size(smooth_set):
    smooth_size_batch = nn.relu(jnp.sum(smooth_set, axis=1) - target_size)
    return jnp.mean(smooth_size_batch)

@jit
def smooth_quantile(x, p):
    return jnp.quantile(x, p) # For now, jax's built-in sample quantile function will do
    #return smooth_quantile(x, p)

@jit
def smooth_score_threshold(params, x, y):
    scores = conformity_score(params, x, y)
    tau = smooth_quantile(scores, alpha * (1 + 1/len(scores)))
    return tau

@jit
def base_loss(params, x, y):
    logits,_ = model.apply(params, x, mutable= ["batch_stats"])
    probabilities = nn.softmax(logits)
    y_encoded = jax.nn.one_hot(y, num_classes)
    return -jnp.sum(y_encoded * jax.nn.log_softmax(probabilities)) / len(x)

@jit
def regularizer(params):
    return sum(jnp.sum(jnp.square(param)) for param in jax.tree.leaves(params)) # L2 regularization

def loss_transform_fn(s):
    return jnp.log(s + 1e-8)

@jit
def coverage_loss(smooth_set, y):
    # shapes: smooth_set (batch_size, num_labels), y (batch_size, )    
    y_encoded = jax.nn.one_hot(y, num_classes)
    l1 = (1 - smooth_set) * y_encoded * loss_matrix[y]
    l2 = smooth_set * (1 - y_encoded) * loss_matrix[y]
    loss = jnp.sum(jnp.maximum(l1 + l2, jnp.zeros_like(l1)), axis=1)
    return jnp.mean(loss)

@jit
def conftr_loss(params, x, y): # Todo: add a flag for coverage_loss
    #! we'll assume that the batch size is even
    x_pred, x_calib = jnp.split(x, 2)
    y_pred, y_calib = jnp.split(y, 2)
    
    tau = smooth_score_threshold(params, x_calib, y_calib)
    return base_loss_weight  * base_loss(params, x, y) \
        + regularizer_weight * regularizer(params) \
        + jnp.log(
            coverage_weight * coverage_loss(smooth_predict_set(params, x_pred, tau), y_pred) \
            + size_weight   * smooth_size(smooth_predict_set(params, x_pred, tau)) \
            + 1e-8
        )

def conftr_loss_partial(params, tau, batch):
    x, y = batch
    return coverage_weight * coverage_loss(smooth_predict_set(params, x, tau), y) \
           + size_weight   * smooth_size(smooth_predict_set(params, x, tau)) 

def conftr_loss_partial_grad_params(param, tau, batch):
    # in julia: gradient(θ -> S(θ, τ, B), θ)
    return jax.grad(lambda param: conftr_loss_partial(param, tau, batch))(param)

def conftr_loss_partial_grad_tau(param, tau, batch):
    # in julia: gradient(τ -> S(θ, τ, B), τ)
    return jax.grad(lambda tau: conftr_loss_partial(param, tau, batch))(tau)

def conftr_loss_partial_grad(param, tau, tau_grad, batch):
    # ∂/∂θ S(θ, τ, B) ≈ ∂S_∂θ(θ, τ̂, B) + ∂S_∂τ(θ, τ̂, B) ∇τ̂    
    return tree_sum(
            conftr_loss_partial_grad_params(param, tau, batch), \
            tree_scale(tau_grad, conftr_loss_partial_grad_tau(param, tau, batch))
            )
'''def evaluate_conformal_prediction(model, params, cal_loader, test_loader, num_trials=10, alpha=0.01):
    
    coverage_rates = []
    average_sizes = []

    cal_dataset = cal_loader.dataset
    test_dataset = test_loader.dataset

    # Combine the datasets
    combined_dataset = ConcatDataset([cal_dataset, test_dataset])

    # Split the cal and test data
    cal_size = len(cal_dataset)
    test_size = len(test_dataset)

    for trial in range(num_trials):
        
        indices = np.random.permutation(len(combined_dataset)).tolist()  # Numpy to randomize indices
        cal_indices = indices[:cal_size]
        test_indices = indices[cal_size:cal_size + test_size]

        cal_subset = Subset(combined_dataset, cal_indices)
        test_subset = Subset(combined_dataset, test_indices)

        cal_loader_shuffled = DataLoader(cal_subset, batch_size=cal_loader.batch_size, shuffle=False)
        test_loader_shuffled = DataLoader(test_subset, batch_size=test_loader.batch_size, shuffle=False)

        ##### Calibrate
        cal_outputs = []
        cal_labels = []
        for inputs, labels in cal_loader_shuffled:
            inputs = jnp.array(inputs)
            labels = jnp.array(labels)
            
            # Get the model output for calibration data using JAX model
            logits, _ = model.apply(params, inputs)
            cal_outputs.append(logits)
            cal_labels.append(labels)
        
        cal_outputs = jnp.concatenate(cal_outputs)
        cal_labels = jnp.concatenate(cal_labels)

        # Non-conformity score is 1 - f(x) of the true y class for the calibration data
        cal_nonconformity_scores = 1 - jnp.take_along_axis(cal_outputs, cal_labels[:, None], axis=1).squeeze()

        # Calculate the 1 - alpha quantile of the non-conformity scores
        quantile = jnp.quantile(cal_nonconformity_scores, 1 - alpha)

        ##### Predict
        test_outputs = []
        test_labels = []
        for inputs, labels in test_loader_shuffled:
            inputs = jnp.array(inputs)
            labels = jnp.array(labels)
            
            # Get the model output for test data using JAX model
            logits, _ = model.apply(params, inputs)
            test_outputs.append(logits)
            test_labels.append(labels)

        test_outputs = jnp.concatenate(test_outputs)
        test_labels = jnp.concatenate(test_labels)

        # Construct a conformal prediction set with all classes > 1 - alpha threshold value
        prediction_sets = (test_outputs >= 1 - quantile)

        # Calculate the coverage rate on the test data
        correct_predictions = jnp.take_along_axis(prediction_sets, test_labels[:, None], axis=1).squeeze()
        coverage_rate = jnp.mean(correct_predictions)

        # Calculate the average prediction set size on the test data
        average_size = jnp.mean(jnp.sum(prediction_sets, axis=1))

        coverage_rates.append(coverage_rate)
        average_sizes.append(average_size)

        print(f"Trial {trial + 1} -- Coverage_rate: {coverage_rate}, Average_size: {average_size}")

    avg_coverage_rate = np.mean(coverage_rates)
    std_coverage_rate = np.std(coverage_rates)
    avg_average_size = np.mean(average_sizes)
    std_average_size = np.std(average_sizes)

    return avg_coverage_rate, std_coverage_rate, avg_average_size, std_average_size

def compute_accuracy(params, data_loader):
    correct_predictions = 0
    total_predictions = 0
    
    for x, y in data_loader:
        x = jnp.array(x)
        y = jnp.array(y)
        
        # Get the logits (raw scores) from the model
        logits = model.apply(params, x)
        
        # Predicted label is the one with the highest score
        predictions = jnp.argmax(logits, axis=1)
        
        # Count the number of correct predictions
        correct_predictions += jnp.sum(predictions == y)
        total_predictions += len(y)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy
'''
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
        epochs = config.epochs
        lr_scheduler = MultIStepLRScheduler(
            learning_rate=config.lr_scheduler.learning_rate,
            learning_rate_decay=config.lr_scheduler.learning_rate_decay,
            num_examples=len(train_loader.dataset),
            batch_size=config.batch_size,
            epochs=epochs
        )
        optimizer = optax.chain(
            optax.trace(decay=config.optimizer_decay, nesterov=True),
            optax.scale_by_schedule(lambda step: -lr_scheduler(step))
        )
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

                #Split into Calibration and Prediction 
                x_pred, x_calib = jnp.split(x, 2)
                y_pred, y_calib = jnp.split(y, 2)

                # Compute tau_hat on the calibration batch
                tau_hat = smooth_score_threshold(params, x_calib, y_calib)

                #Splitting Procedure
                n = 5
                batch_size = x_calib.shape[0] // n
                tau_grad_hat = tree_zero(params) #initialization
                for idx in range(n):
                    start_idx = idx  * batch_size
                    end_idx = start_idx + batch_size

                    x_calib_mb = x_calib[start_idx:end_idx]
                    y_calib_mb = y_calib[start_idx:end_idx]

                    #calcualte the grad of tau on each minibatch
                    tau_grad_hat = tree_sum(tau_grad_hat, jax.grad(smooth_score_threshold)(params, x_calib_mb, y_calib_mb))
                
                #average the gradients
                tau_grad_hat = tree_scale(tau_grad_hat, 1/n)

                #compute the conftr loss with the tau and the grad tau computed above. 
                s_eval = conftr_loss_partial(params, tau_hat, (x_pred, y_pred))

                grad_s = conftr_loss_partial_grad(params, tau_hat, tau_grad_hat, (x_pred, y_pred))
            
                base_grad = jax.grad(base_loss)(params, x, y)

                regularizer_grad = jax.grad(regularizer)(params)


                #not sure about this part, just copied from fedconftr but not sure about the scale and sum ? 
                grad = tree_zero(params)
                grad = tree_sum(grad, tree_scale(grad_s, jax.grad(loss_transform_fn)(s_eval)))
                grad = tree_sum(grad, tree_scale(base_grad, base_loss_weight))
                grad = tree_sum(grad, tree_scale(regularizer_grad, regularizer_weight))
                ################## previous loooppppp ################
                # update the params
                update, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, update)


                # Compute loss statistics
                loss = conftr_loss(params, x, y)
                batch_losses.append(loss.item())
                running_loss += loss.item()

                #End of Epoch
            
            # Statistics Calculations at the end of Epoch
            epoch_train_loss = running_loss/len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_train_loss:.4f}")

            # Evaluate Loss on the Test Set
            epoch_test_loss = jnp.mean(
                jnp.array([
                    conftr_loss(params, jnp.array(x_test), jnp.array(y_test))
                    for x_test, y_test in test_loader
                ]))
            print(f"Epoch [{epoch+1}/{epochs}] Test Loss: {epoch_test_loss:.4f}")

            # Evaluate Accuracy on the test Set
            test_accuracy = compute_accuracy(model, params, test_loader)

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






            

            








