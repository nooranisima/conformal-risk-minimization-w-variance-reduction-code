from torch.utils.data import ConcatDataset, Subset, DataLoader
from models.models import get_model
from data.data_loader import load_data
import torch
import numpy as np 
import jax.numpy as jnp


def evaluate_conformal_prediction(model, params, cal_loader, test_loader, num_trials=10, alpha=0.01):
    
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
            
            # Get the model output for calibration data 
            logits = model.apply(params, inputs)
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
            
            # Get the model output for test data 
            logits = model.apply(params, inputs)
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

def compute_accuracy(model, params, test_loader):
    correct_predictions = 0
    total_predictions = 0
    
    # Iterate through the test_loader
    for inputs, labels in test_loader:
        # Convert to JAX arrays
        inputs = jnp.array(inputs)
        labels = jnp.array(labels)
        
        # Get the model predictions (logits) using the current parameters
        logits= model.apply(params, inputs)
        
        # Get the predicted class by finding the argmax along the logits
        predictions = jnp.argmax(logits, axis=1)
        
        # Count correct predictions
        correct_predictions += jnp.sum(predictions == labels)
        
        # Count the total number of predictions
        total_predictions += len(labels)
    
    # Calculate accuracy as correct predictions divided by total predictions
    accuracy = correct_predictions / total_predictions
    
    return accuracy


def main(models_info):
    accuracies = []
    results = {}

    for (model_name, model_config, (model_params, seed)) in models_info:
        model_config = model_config()

        _, calib_loader, test_loader = load_data(model_config, seed)

        model = get_model(model_config)

        #Compute Model accuracy
        accuracy = compute_accuracy(model, model_params, test_loader)
        accuracies.append(accuracy)
        
        # Evaluate THR CP
        avg_coverage_rate, std_coverage_rate, avg_average_size, std_average_size = evaluate_conformal_prediction(
            model, model_params, calib_loader, test_loader, num_trials=10, alpha=0.01)

        print(f"Model Conformal Prediction - Avg Coverage Rate: {avg_coverage_rate}, Std Coverage Rate: {std_coverage_rate}, Avg Size: {avg_average_size}, Std Size: {std_average_size}")

        results[model_name] = {
                'accuracy': accuracy,
                'avg_coverage_rate': avg_coverage_rate,
                'std_coverage_rate': std_coverage_rate,
                'avg_size': avg_average_size,
                'std_size': std_average_size
            }
    return results

