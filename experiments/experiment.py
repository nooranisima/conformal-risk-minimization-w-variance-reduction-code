import os
from config import *
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config.config import get_experiment_config_mnist, get_experiment_config_fmnist
from data.data_loader import load_data
from algorithms.vr_split import main as vr_split_main
from algorithms.vr_sort import main as vr_sort_main
from algorithms.conftr import main as conftr_main
from algorithms.baseline import main as base_main
from evaluation import main as eval_main

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

############# HELPER FUNCTIONS ###############
def plot_combined_losses(*loss_data, title, save_path, loss_type='Train'):
    """
    Plots multiple sets of loss data with their corresponding standard deviations.
    
    Parameters:
        *loss_data (tuple): Each tuple should contain (losses, stds, label, color).
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
        loss_type (str): Type of loss, e.g., 'Train' or 'Test'.
    """
    plt.figure()
    
    # Iterate over all loss data
    for losses, std, label, color in loss_data:
        plt.plot(losses, label = label, color = color, linewidth = 2)
        plt.fill_between(range(len(losses)),
                            np.array(losses) - np.array(std),
                            np.array(losses) + np.array(std),
                            color = color, alpha = 0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_set_sizes(*size_data, title, save_path):
    """
    Plots set sizes over epochs for multiple models.
    
    Parameters:
        *size_data (tuple): Each tuple should contain (set_sizes, label, color).
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    plt.figure()

    # Iterate over all size data
    for set_sizes, std, label, color in size_data:
        plt.plot(set_sizes, label=label, color=color, linewidth=2)
        plt.fill_between(range(len(set_sizes)),
                            np.array(set_sizes) - np.array(std),
                            np.array(set_sizes) + np.array(std),
                            color = color, alpha = 0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Set Size')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

def plot_accuracies(*accuracy_data, title, save_path):
    """
    Plots accuracies over epochs for multiple models.
    
    Parameters:
        *accuracy_data (tuple): Each tuple should contain (accuracies, label, color).
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    plt.figure()

    # Iterate over all accuracy data
    for accuracies, std,  label, color in accuracy_data:
        plt.plot(accuracies, label=label, color=color, linewidth=2)
        plt.fill_between(range(len(accuracies)),
                            np.array(accuracies) - np.array(std),
                            np.array(accuracies) + np.array(std),
                            color = color, alpha = 0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

def plot_avg_sizes(*model_data, title, save_path):
    """
    Plots average sizes against model names.
    
    Parameters:
        *model_data (tuple): Each tuple should contain (model_name, avg_size).
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    model_names = [data[0] for data in model_data]
    avg_sizes = [data[1] for data in model_data]
    
    plt.figure()
    
    plt.bar(model_names, avg_sizes, color='skyblue')
    plt.xlabel('Model Name')
    plt.ylabel('Average Size')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

def average_results(results_list):
    avg_results = {}
    for key in results_list[0]:
        avg_results[key] = np.mean([result[key] for result in results_list])
    return avg_results

def save_model(params, model_name, results_dir):
    model_save_path = os.path.join(results_dir, model_name)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(params, f)

################ TRAINING ################
def plot_tuning(results_dir, old_results_dir):

    experiment_results_path = os.path.join(old_results_dir, 'experiment_results_sort.pkl')
    with open(experiment_results_path, 'rb') as f:
        sort_results = pickle.load(f)
    results_dir = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_sort_values = [4, 6, 8, 10, 16, 20]

    color_mapping = {
    4: 'red',
    6: 'blue',
    8: 'green',
    10: 'orange',
    16: 'purple',
    20: 'yellow',
    }

    # Plot Training and Test Losses for different `num_splits`
    for loss_type in ['Train', 'Test']:
        plot_combined_losses(
            *[
                (sort_results[num_sort][f'{loss_type.lower()}_losses'], 
                 sort_results[num_sort][f'{loss_type.lower()}_std'], 
                 f'VR-ConfTr {loss_type} loss (m={num_sort})', 
                 color_mapping[num_sort])  # Generate random colors for each num_sort
                for num_sort in num_sort_values
            ],
            title=f'{loss_type} Loss per Epoch for Different m',
            save_path=os.path.join(results_dir, f'combined_{loss_type.lower()}_losses_sort.png'),
            loss_type=loss_type
        )

    # Plot Set Sizes for different `num_splits`
    plot_set_sizes(
        *[
            (sort_results[num_sort]['set_sizes'], 
             sort_results[num_sort]['set_std'], 
             f'VR-ConfTr Model Set Sizes (m={num_sort})', 
             color_mapping[num_sort])
            for num_sort in num_sort_values
        ],
        title='Set Sizes per Epoch for Different m',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch_sort.png')
    )

    # Plot Test Accuracy per Epoch during training for different `num_splits`
    plot_accuracies(
        *[
            (sort_results[num_sort]['test_accuracies'], 
             sort_results[num_sort]['test_accuracies_std'], 
             f'VR-ConfTr Test Accuracy (m={num_sort})', 
             color_mapping[num_sort])
            for num_sort in num_sort_values
        ],
        title='Test Accuracy per Epoch for Different m',
        save_path=os.path.join(results_dir, 'combined_test_accuracies_sort.png')
    )


def run_tuning(config_func, results_dir):
    config = get_experiment_config_fmnist() #TODO: figure out to pass the right config format TypeError: 'ConfigDict' object is not callable
    num_trials = 2
    results_dir = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Storing Results 
    experiment_results = {}

    #List of the grid search for the split tuning
    num_sort_values = [4, 6, 8, 10, 16, 20]

    color_mapping = {
    4: 'red',
    6: 'blue',
    8: 'green',
    10: 'orange',
    16: 'purple',
    20: 'yellow',
    }

    #Place Holder for results
    sort_results = {}



    for num_sort in num_sort_values:
        print(f"Running experiment with num_splits={num_sort}")

        sort_trial_params_and_seeds,(sort_train_losses,  sort_std_train_losses),(sort_test_losses, sort_std_test_losses),(sort_test_accuracies, sort_std_test_accuracies), sort_loss_variances, (sort_set_sizes, sort_std_set_sizes) = vr_sort_main(config.vr, num_trials, num_sort = num_sort)
        save_model(sort_trial_params_and_seeds, config.sort_model_path, results_dir)

        sort_results[num_sort] = {
                'train_losses': sort_train_losses,
                'train_std': sort_std_train_losses,
                'test_losses': sort_test_losses,
                'test_std': sort_std_test_losses,
                'test_accuracies': sort_test_accuracies,
                'test_accuracies_std': sort_std_test_accuracies,
                'loss_variances': sort_loss_variances,
                'set_sizes': sort_set_sizes,
                'set_std': sort_std_set_sizes
            }

    # Plot Training and Test Losses for different `num_splits`
    for loss_type in ['Train', 'Test']:
        plot_combined_losses(
            *[
                (sort_results[num_sort][f'{loss_type.lower()}_losses'], 
                 sort_results[num_sort][f'{loss_type.lower()}_std'], 
                 f'VR-CT {loss_type} loss (m={num_sort})', 
                 color_mapping[num_sort]
                )  # Generate random colors for each num_sort
                for num_sort in num_sort_values
            ],
            title=f'{loss_type} Loss per Epoch for Different num_sort',
            save_path=os.path.join(results_dir, f'combined_{loss_type.lower()}_losses_sort.png'),
            loss_type=loss_type
        )

    # Plot Set Sizes for different `num_splits`
    plot_set_sizes(
        *[
            (sort_results[num_sort]['set_sizes'], 
             sort_results[num_sort]['set_std'], 
             f'VR-CT Model Set Sizes (m={num_sort})', 
             color_mapping[num_sort])
            for num_sort in num_sort_values
        ],
        title='Set Sizes per Epoch for Different num_sort',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch_sort.png')
    )

    # Plot Test Accuracy per Epoch during training for different `num_splits`
    plot_accuracies(
        *[
            (sort_results[num_sort]['test_accuracies'], 
             sort_results[num_sort]['test_accuracies_std'], 
             f'VR-CT Test Accuracy (m={num_sort})', 
             color_mapping[num_sort])
            for num_sort in num_sort_values
        ],
        title='Test Accuracy per Epoch for Different num_sort',
        save_path=os.path.join(results_dir, 'combined_test_accuracies_sort.png')
    )

    # Save all experiment results in the results directory 
    experiment_results_path = os.path.join(results_dir, 'experiment_results_sort.pkl')
    with open(experiment_results_path, 'wb') as f:
        pickle.dump(sort_results, f)

    print(f"Experiment results saved at {experiment_results_path}")

def run_experiment_mnist():
    config = get_experiment_config_mnist()
    num_trials = config.num_trials
    results_dir = config.results_dir
    print("results", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    experiment_results = {}

    #Baseline Model Training
    print("Baseline ")
    base_trial_params_and_seeds,(base_train_losses,  base_std_train_losses),(base_test_losses, base_std_test_losses),(base_test_accuracies, base_std_test_accuracies), base_loss_variances, (base_set_sizes, base_std_set_sizes) = base_main(config.vr, num_trials)
    save_model(base_trial_params_and_seeds, config.base_model_path, results_dir)
    experiment_results['base'] = {
        'model_params': base_trial_params_and_seeds,
        'train_losses': base_train_losses,
        'train_std': base_std_train_losses,
        'test_losses': base_test_losses,
        'test_std': base_std_test_losses,
        'test_accuracies': base_test_accuracies,
        'test_accuracies_std': base_std_test_accuracies,
        'loss_variances': base_loss_variances,
        'set_sizes': base_set_sizes,
        'set_std' : base_std_set_sizes
    }

    
    print("Sorting Version")
    sort_trial_params_and_seeds,(sort_train_losses,  sort_std_train_losses),(sort_test_losses, sort_std_test_losses),(sort_test_accuracies, sort_std_test_accuracies), sort_loss_variances, (sort_set_sizes, sort_std_set_sizes) = vr_sort_main(config.vr, num_trials, num_sort = 6)
    save_model(sort_trial_params_and_seeds, config.sort_model_path, results_dir)
    experiment_results['sort'] = {
        'model_params': sort_trial_params_and_seeds,
        'train_losses': sort_train_losses,
        'train_std': sort_std_train_losses,
        'test_losses': sort_test_losses,
        'test_std': sort_std_test_losses,
        'test_accuracies': sort_test_accuracies,
        'test_accuracies_std': sort_std_test_accuracies,
        'loss_variances': sort_loss_variances,
        'set_sizes': sort_set_sizes,
        'set_std' : sort_std_set_sizes
    }

    print("Conftr Version")
    #Conftr Model Training
    conftr_trial_params_and_seeds,(conftr_train_losses,  conftr_std_train_losses),(conftr_test_losses, conftr_std_test_losses),(conftr_test_accuracies, conftr_std_test_accuracies), conftr_loss_variances, (conftr_set_sizes, conftr_std_set_sizes) = conftr_main(config.vr, num_trials)
    save_model(conftr_trial_params_and_seeds, config.conftr_model_path, results_dir)
    experiment_results['conftr'] = {
        'model_params': conftr_trial_params_and_seeds,
        'train_losses': conftr_train_losses,
        'train_std': conftr_std_train_losses,
        'test_losses': conftr_test_losses,
        'test_std': conftr_std_test_losses,
        'test_accuracies': conftr_test_accuracies,
        'test_accuracies_std': conftr_std_test_accuracies,
        'loss_variances': conftr_loss_variances,
        'set_sizes': conftr_set_sizes,
        'set_std' : conftr_std_set_sizes
    }



    '''print("Splitting Version")
    split_trial_params_and_seeds,(split_train_losses,  split_std_train_losses),(split_test_losses, split_std_test_losses),(split_test_accuracies, split_std_test_accuracies), split_loss_variances, (split_set_sizes, split_std_set_sizes) = vr_split_main(config.vr, num_trials)
    save_model(split_trial_params_and_seeds, config.split_model_path, results_dir)
    experiment_results['split'] = {
        'model_params': split_trial_params_and_seeds,
        'train_losses': split_train_losses,
        'train_std': split_std_train_losses,
        'test_losses': split_test_losses,
        'test_std': split_std_test_losses,
        'test_accuracies': split_test_accuracies,
        'test_accuracies_std': split_std_test_accuracies,
        'loss_variances': split_loss_variances,
        'set_sizes': split_set_sizes,
        'set_std' : split_std_set_sizes
    }'''

    # Plot Train Losses
    plot_combined_losses(
        (sort_train_losses, sort_std_train_losses, 'VR Train loss', 'orange'),
        #(split_train_losses, split_std_train_losses, 'split VR Train loss', 'red'),
        (conftr_train_losses, conftr_std_train_losses, 'Conftr Train loss', 'blue'),
        title = 'Training Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_train_losses.png'),
        loss_type = 'Train'
    )

    # Plot Test loss per epoch 
    plot_combined_losses(
        (sort_test_losses, sort_std_test_losses, 'VR Test loss', 'orange'),
        #(split_test_losses, split_std_test_losses, 'Split VR Test loss', 'red'),
        (conftr_test_losses, conftr_std_test_losses, 'Conftr Test loss', 'blue'),
        title = 'Test Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_test_losses.png'),
        loss_type = 'Test'
    )

    # Plot Set Sizes per epoch
    plot_set_sizes(
        (sort_set_sizes, sort_std_set_sizes, 'VR Model Set Sizes', 'orange'),
        #(split_set_sizes, split_std_set_sizes, 'Split Model Set Sizes', 'red'),
        (conftr_set_sizes, conftr_std_set_sizes, 'Conftr Model Set Sizes', 'blue'),
        (base_set_sizes, base_std_set_sizes, 'Baseline Model Set Sizes', 'green'),
        title='Set Sizes per Epoch',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch.png')
    )

    # Plot accuracies per epoch
    plot_accuracies(
        (sort_test_accuracies, sort_std_test_accuracies, 'VR Test Accuracy', 'orange'),
        #(split_test_accuracies, split_std_test_accuracies, 'Split Test Accuracy', 'red'),
        (conftr_test_accuracies, conftr_std_test_accuracies, 'Conftr Test Accuracy', 'blue'),
        (base_test_accuracies, base_std_test_accuracies, 'Baseline Test Accuracy', 'green'),
        title='Training Accuracy per Epoch',
        save_path=os.path.join(results_dir, 'combined_train_accuracies.png')
    )

    # Evaluate the models on their corresponding test_loader 
    all_results = []
    #split_results = []
    sort_results = []
    conftr_results = []
    base_results = []

    for trial in range(num_trials):
        results = eval_main( models_info=[
                            ("sort_model", config.vr, sort_trial_params_and_seeds[trial]),
                            #("split_model", config.vr_lclass, split_trial_params_and_seeds[trial]),
                            ("conftr_model", config.vr, conftr_trial_params_and_seeds[trial]),
                            ("base_model", config.vr, base_trial_params_and_seeds[trial])
                ])
        all_results.append(results)
        #split_results.append(results["split_model"])
        sort_results.append(results["sort_model"])
        conftr_results.append(results["conftr_model"])
        base_results.append(results["base_model"])

    #avg_split_results = average_results(split_results)
    avg_sort_results = average_results(sort_results)
    avg_conftr_results = average_results(conftr_results)
    avg_base_results = average_results(base_results)

    # Add evaluation results to the experiment results dictionary
    experiment_results['evaluation'] = {
        'all_results': all_results,
        #'split_results': split_results,
        'conftr_results': conftr_results,
        'base_results': base_results,
        'sort_results': sort_results
    }
    experiment_results['avg_results'] = {
        #'split': avg_split_results,
        'conftr': avg_conftr_results,
        'base': avg_base_results,
        'sort': avg_sort_results
    }

    # Plot histogram of average sizes
    avg_sizes = { #"split Model" : avg_split_results['avg_size'],
                  "Conftr Model" : avg_conftr_results['avg_size'],
                  "Base Model" : avg_base_results['avg_size'],
                  "Sort Model" : avg_sort_results['avg_size']
    }


    plot_avg_sizes(
        *[(name, size) for name, size in avg_sizes.items()],
        title='Average Sizes of Models',
        save_path=os.path.join(results_dir, 'avg_sizes_plot.png')
    )

    # Save all experiment results in the results directory 
    experiment_results_path = os.path.join(results_dir, 'experiment_results.pkl')
    with open(experiment_results_path, 'wb') as f:
        pickle.dump(experiment_results, f)

def run_experiment_fmnist():
    config = get_experiment_config_fmnist()
    num_trials = config.num_trials
    results_dir = config.results_dir
    print("results", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    experiment_results = {}

    #Baseline Model Training
    '''print("Baseline ")
    base_trial_params_and_seeds,(base_train_losses,  base_std_train_losses),(base_test_losses, base_std_test_losses),(base_test_accuracies, base_std_test_accuracies), base_loss_variances, (base_set_sizes, base_std_set_sizes) = base_main(config.vr, num_trials)
    save_model(base_trial_params_and_seeds, config.base_model_path, results_dir)
    experiment_results['base'] = {
        'model_params': base_trial_params_and_seeds,
        'train_losses': base_train_losses,
        'train_std': base_std_train_losses,
        'test_losses': base_test_losses,
        'test_std': base_std_test_losses,
        'test_accuracies': base_test_accuracies,
        'test_accuracies_std': base_std_test_accuracies,
        'loss_variances': base_loss_variances,
        'set_sizes': base_set_sizes,
        'set_std' : base_std_set_sizes
    }'''

    
    print("Sorting Version")
    sort_trial_params_and_seeds,(sort_train_losses,  sort_std_train_losses),(sort_test_losses, sort_std_test_losses),(sort_test_accuracies, sort_std_test_accuracies), sort_loss_variances, (sort_set_sizes, sort_std_set_sizes) = vr_sort_main(config.vr, num_trials, num_sort = 6)
    save_model(sort_trial_params_and_seeds, config.sort_model_path, results_dir)
    experiment_results['sort'] = {
        'model_params': sort_trial_params_and_seeds,
        'train_losses': sort_train_losses,
        'train_std': sort_std_train_losses,
        'test_losses': sort_test_losses,
        'test_std': sort_std_test_losses,
        'test_accuracies': sort_test_accuracies,
        'test_accuracies_std': sort_std_test_accuracies,
        'loss_variances': sort_loss_variances,
        'set_sizes': sort_set_sizes,
        'set_std' : sort_std_set_sizes
    }

    print("Conftr Version")
    #Conftr Model Training
    conftr_trial_params_and_seeds,(conftr_train_losses,  conftr_std_train_losses),(conftr_test_losses, conftr_std_test_losses),(conftr_test_accuracies, conftr_std_test_accuracies), conftr_loss_variances, (conftr_set_sizes, conftr_std_set_sizes) = conftr_main(config.vr, num_trials)
    save_model(conftr_trial_params_and_seeds, config.conftr_model_path, results_dir)
    experiment_results['conftr'] = {
        'model_params': conftr_trial_params_and_seeds,
        'train_losses': conftr_train_losses,
        'train_std': conftr_std_train_losses,
        'test_losses': conftr_test_losses,
        'test_std': conftr_std_test_losses,
        'test_accuracies': conftr_test_accuracies,
        'test_accuracies_std': conftr_std_test_accuracies,
        'loss_variances': conftr_loss_variances,
        'set_sizes': conftr_set_sizes,
        'set_std' : conftr_std_set_sizes
    }



    '''print("Splitting Version")
    split_trial_params_and_seeds,(split_train_losses,  split_std_train_losses),(split_test_losses, split_std_test_losses),(split_test_accuracies, split_std_test_accuracies), split_loss_variances, (split_set_sizes, split_std_set_sizes) = vr_split_main(config.vr, num_trials)
    save_model(split_trial_params_and_seeds, config.split_model_path, results_dir)
    experiment_results['split'] = {
        'model_params': split_trial_params_and_seeds,
        'train_losses': split_train_losses,
        'train_std': split_std_train_losses,
        'test_losses': split_test_losses,
        'test_std': split_std_test_losses,
        'test_accuracies': split_test_accuracies,
        'test_accuracies_std': split_std_test_accuracies,
        'loss_variances': split_loss_variances,
        'set_sizes': split_set_sizes,
        'set_std' : split_std_set_sizes
    }'''

    # Plot Train Losses
    plot_combined_losses(
        (sort_train_losses, sort_std_train_losses, 'VR Train loss', 'orange'),
        #(split_train_losses, split_std_train_losses, 'split VR Train loss', 'red'),
        (conftr_train_losses, conftr_std_train_losses, 'Conftr Train loss', 'blue'),
        title = 'Training Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_train_losses.png'),
        loss_type = 'Train'
    )

    # Plot Test loss per epoch 
    plot_combined_losses(
        (sort_test_losses, sort_std_test_losses, 'VR Test loss', 'orange'),
        #(split_test_losses, split_std_test_losses, 'Split VR Test loss', 'red'),
        (conftr_test_losses, conftr_std_test_losses, 'Conftr Test loss', 'blue'),
        title = 'Test Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_test_losses.png'),
        loss_type = 'Test'
    )

    # Plot Set Sizes per epoch
    plot_set_sizes(
        (sort_set_sizes, sort_std_set_sizes, 'VR Model Set Sizes', 'orange'),
        #(split_set_sizes, split_std_set_sizes, 'Split Model Set Sizes', 'red'),
        (conftr_set_sizes, conftr_std_set_sizes, 'Conftr Model Set Sizes', 'blue'),
        #(base_set_sizes, base_std_set_sizes, 'Baseline Model Set Sizes', 'green'),
        title='Set Sizes per Epoch',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch.png')
    )

    # Plot accuracies per epoch
    plot_accuracies(
        (sort_test_accuracies, sort_std_test_accuracies, 'VR Test Accuracy', 'orange'),
        #(split_test_accuracies, split_std_test_accuracies, 'Split Test Accuracy', 'red'),
        (conftr_test_accuracies, conftr_std_test_accuracies, 'Conftr Test Accuracy', 'blue'),
        #(base_test_accuracies, base_std_test_accuracies, 'Baseline Test Accuracy', 'green'),
        title='Training Accuracy per Epoch',
        save_path=os.path.join(results_dir, 'combined_train_accuracies.png')
    )

    # Evaluate the models on their corresponding test_loader 
    all_results = []
    #split_results = []
    sort_results = []
    conftr_results = []
    #base_results = []

    for trial in range(num_trials):
        results = eval_main( models_info=[
                            ("sort_model", config.vr, sort_trial_params_and_seeds[trial]),
                            #("split_model", config.vr, split_trial_params_and_seeds[trial]),
                            ("conftr_model", config.vr, conftr_trial_params_and_seeds[trial]),
                            #("base_model", config.vr, base_trial_params_and_seeds[trial])
                ])
        all_results.append(results)
        #split_results.append(results["split_model"])
        sort_results.append(results["sort_model"])
        conftr_results.append(results["conftr_model"])
        #base_results.append(results["base_model"])

    #avg_split_results = average_results(split_results)
    avg_sort_results = average_results(sort_results)
    avg_conftr_results = average_results(conftr_results)
    #avg_base_results = average_results(base_results)

    # Add evaluation results to the experiment results dictionary
    experiment_results['evaluation'] = {
        'all_results': all_results,
        #'split_results': split_results,
        'conftr_results': conftr_results,
        #'base_results': base_results,
        'sort_results': sort_results
    }
    experiment_results['avg_results'] = {
        #'split': avg_split_results,
        'conftr': avg_conftr_results,
        #'base': avg_base_results,
        'sort': avg_sort_results
    }

    # Plot histogram of average sizes
    avg_sizes = { #"split Model" : avg_split_results['avg_size'],
                  "Conftr Model" : avg_conftr_results['avg_size'],
                  #"Base Model" : avg_base_results['avg_size'],
                  "Sort Model" : avg_sort_results['avg_size']
    }


    plot_avg_sizes(
        *[(name, size) for name, size in avg_sizes.items()],
        title='Average Sizes of Models',
        save_path=os.path.join(results_dir, 'avg_sizes_plot.png')
    )

    # Save all experiment results in the results directory 
    experiment_results_path = os.path.join(results_dir, 'experiment_results.pkl')
    with open(experiment_results_path, 'wb') as f:
        pickle.dump(experiment_results, f)

def run_test_fmnist():
    config = get_experiment_config_fmnist()
    num_trials = 1
    results_dir = os.path.join(os.getcwd(), 'Test-Fashion-Mnist')
    print("results", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    experiment_results = {}

    #for batch size of 500, use config.vr, for batch size of 100, use config.ct100

    sort_trial_params_and_seeds,(sort_train_losses,  sort_std_train_losses),(sort_test_losses, sort_std_test_losses),(sort_test_accuracies, sort_std_test_accuracies), sort_loss_variances, (sort_set_sizes, sort_std_set_sizes) = vr_sort_main(config.vr, num_trials, num_sort = 6)
    save_model(sort_trial_params_and_seeds, config.sort_model_path, results_dir)
    experiment_results['sort'] = {
        'model_params': sort_trial_params_and_seeds,
        'train_losses': sort_train_losses,
        'train_std': sort_std_train_losses,
        'test_losses': sort_test_losses,
        'test_std': sort_std_test_losses,
        'test_accuracies': sort_test_accuracies,
        'test_accuracies_std': sort_std_test_accuracies,
        'loss_variances': sort_loss_variances,
        'set_sizes': sort_set_sizes,
        'set_std' : sort_std_set_sizes
    }

    conftr100_trial_params_and_seeds,(conftr100_train_losses,  conftr100_std_train_losses),(conftr100_test_losses, conftr100_std_test_losses),(conftr100_test_accuracies, conftr100_std_test_accuracies), conftr100_loss_variances, (conftr100_set_sizes, conftr100_std_set_sizes) = conftr_main(config.ct100, num_trials)
    save_model(conftr100_trial_params_and_seeds, config.conftr_model_path, results_dir)
    experiment_results['conftr100'] = {
        'model_params': conftr100_trial_params_and_seeds,
        'train_losses': conftr100_train_losses,
        'train_std': conftr100_std_train_losses,
        'test_losses': conftr100_test_losses,
        'test_std': conftr100_std_test_losses,
        'test_accuracies': conftr100_test_accuracies,
        'test_accuracies_std': conftr100_std_test_accuracies,
        'loss_variances': conftr100_loss_variances,
        'set_sizes': conftr100_set_sizes,
        'set_std' : conftr100_std_set_sizes
    }

    conftr500_trial_params_and_seeds,(conftr500_train_losses,  conftr500_std_train_losses),(conftr500_test_losses, conftr500_std_test_losses),(conftr500_test_accuracies, conftr500_std_test_accuracies), conftr500_loss_variances, (conftr500_set_sizes, conftr500_std_set_sizes) = conftr_main(config.vr, num_trials)
    save_model(conftr500_trial_params_and_seeds, config.conftr_model_path, results_dir)
    experiment_results['conftr500'] = {
        'model_params': conftr500_trial_params_and_seeds,
        'train_losses': conftr500_train_losses,
        'train_std': conftr500_std_train_losses,
        'test_losses': conftr500_test_losses,
        'test_std': conftr500_std_test_losses,
        'test_accuracies': conftr500_test_accuracies,
        'test_accuracies_std': conftr500_std_test_accuracies,
        'loss_variances': conftr500_loss_variances,
        'set_sizes': conftr500_set_sizes,
        'set_std' : conftr500_std_set_sizes
    }

    # Plot Train Losses
    plot_combined_losses(
        (sort_train_losses, sort_std_train_losses, 'VR Train loss', 'orange'),
        (conftr500_train_losses, conftr500_std_train_losses, 'Conftr500 Train loss', 'red'),
        (conftr100_train_losses, conftr100_std_train_losses, 'Conftr100 Train loss', 'blue'),
        title = 'Training Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_train_losses.png'),
        loss_type = 'Train'
    )

    # Plot Test loss per epoch 
    plot_combined_losses(
        (sort_test_losses, sort_std_test_losses, 'VR Test loss', 'orange'),
        (conftr500_test_losses, conftr500_std_test_losses, 'Conftr500 Test loss', 'red'),
        (conftr100_test_losses, conftr100_std_test_losses, 'Conftr100 Test loss', 'blue'),
        title = 'Test Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_test_losses.png'),
        loss_type = 'Test'
    )

    # Plot Set Sizes per epoch
    plot_set_sizes(
        (sort_set_sizes, sort_std_set_sizes, 'VR Model Set Sizes', 'orange'),
        (conftr500_set_sizes, conftr500_std_set_sizes, 'Conftr500 Model Set Sizes', 'red'),
        (conftr100_set_sizes, conftr100_std_set_sizes, 'Conftr100 Model Set Sizes', 'blue'),
        #(base_set_sizes, base_std_set_sizes, 'Baseline Model Set Sizes', 'green'),
        title='Set Sizes per Epoch',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch.png')
    )

    # Plot accuracies per epoch
    plot_accuracies(
        (sort_test_accuracies, sort_std_test_accuracies, 'VR Test Accuracy', 'orange'),
        (conftr500_test_accuracies, conftr500_std_test_accuracies, 'Conftr500 Test Accuracy', 'red'),
        (conftr100_test_accuracies, conftr100_std_test_accuracies, 'Conftr100 Test Accuracy', 'blue'),
        #(base_test_accuracies, base_std_test_accuracies, 'Baseline Test Accuracy', 'green'),
        title='Training Accuracy per Epoch',
        save_path=os.path.join(results_dir, 'combined_train_accuracies.png')
    )

    # Evaluate the models on their corresponding test_loader 
    all_results = []
    sort_results = []
    conftr100_results = []
    conftr500_results = []

    for trial in range(num_trials):
        results = eval_main( models_info=[
                            ("sort_model", config.vr, sort_trial_params_and_seeds[trial]),
                            #("split_model", config.vr, split_trial_params_and_seeds[trial]),
                            ("conftr100_model", config.vr, conftr100_trial_params_and_seeds[trial]),
                            ("conftr500_model", config.vr, conftr500_trial_params_and_seeds[trial]),
                            #("base_model", config.vr, base_trial_params_and_seeds[trial])
                ])
        all_results.append(results)
        #split_results.append(results["split_model"])
        sort_results.append(results["sort_model"])
        conftr100_results.append(results["conftr100_model"])
        conftr500_results.append(results["conftr500_model"])
        #base_results.append(results["base_model"])
    avg_sort_results = average_results(sort_results)
    avg_conftr100_results = average_results(conftr100_results)
    avg_conftr500_results = average_results(conftr500_results)


    # Add evaluation results to the experiment results dictionary
    experiment_results['evaluation'] = {
        'all_results': all_results,
        #'split_results': split_results,
        'conftr100_results': conftr100_results,
        'conftr500_results': conftr500_results,
        #'base_results': base_results,
        'sort_results': sort_results
    }
    experiment_results['avg_results'] = {
        #'split': avg_split_results,
        'conftr100': avg_conftr100_results,
        'conftr500': avg_conftr500_results,
        #'base': avg_base_results,
        'sort': avg_sort_results
    }

    avg_sizes = { #"split Model" : avg_split_results['avg_size'],
                  "Conftr100 Model" : avg_conftr100_results['avg_size'],
                  "Conftr500 Model" : avg_conftr500_results['avg_size'],
                  #"Base Model" : avg_base_results['avg_size'],
                  "Sort Model" : avg_sort_results['avg_size']
    }

    # Save all experiment results in the results directory 
    experiment_results_path = os.path.join(results_dir, 'experiment_results.pkl')
    with open(experiment_results_path, 'wb') as f:
        pickle.dump(experiment_results, f)


def run_baseline_test(dataset = "mnist"):
    if dataset == "mnist":
        config = get_experiment_config_mnist()
        results_dir = os.path.join(os.getcwd(), 'Test-Baseline-MNIST')
    elif dataset == 'fmnist':
        config = get_experiment_config_fmnist()
        results_dir = os.path.join(os.getcwd(), 'Test-Baseline-Fashion-MNIST')

    experiment_results = {}
    
    num_trials = 1
    #Baseline Model Training
    print("Baseline ")
    base_trial_params_and_seeds,(base_train_losses,  base_std_train_losses),(base_test_losses, base_std_test_losses),(base_test_accuracies, base_std_test_accuracies), base_loss_variances, (base_set_sizes, base_std_set_sizes) = base_main(config.vr, num_trials)
    save_model(base_trial_params_and_seeds, config.base_model_path, results_dir)
    experiment_results['base'] = {
        'model_params': base_trial_params_and_seeds,
        'train_losses': base_train_losses,
        'train_std': base_std_train_losses,
        'test_losses': base_test_losses,
        'test_std': base_std_test_losses,
        'test_accuracies': base_test_accuracies,
        'test_accuracies_std': base_std_test_accuracies,
        'loss_variances': base_loss_variances,
        'set_sizes': base_set_sizes,
        'set_std' : base_std_set_sizes
    }
    print("ConfTr")
    conftr_trial_params_and_seeds,(conftr_train_losses,  conftr_std_train_losses),(conftr_test_losses, conftr_std_test_losses),(conftr_test_accuracies, conftr_std_test_accuracies), conftr_loss_variances, (conftr_set_sizes, conftr_std_set_sizes) = conftr_main(config.vr, num_trials)
    save_model(conftr_trial_params_and_seeds, config.conftr_model_path, results_dir)
    experiment_results['conftr'] = {
        'model_params': conftr_trial_params_and_seeds,
        'train_losses': conftr_train_losses,
        'train_std': conftr_std_train_losses,
        'test_losses': conftr_test_losses,
        'test_std': conftr_std_test_losses,
        'test_accuracies': conftr_test_accuracies,
        'test_accuracies_std': conftr_std_test_accuracies,
        'loss_variances': conftr_loss_variances,
        'set_sizes': conftr_set_sizes,
        'set_std' : conftr_std_set_sizes
    }
    
    # Plot Train Losses
    plot_combined_losses(
        (sort_train_losses, sort_std_train_losses, 'VR Train loss', 'orange'),
        #(split_train_losses, split_std_train_losses, 'split VR Train loss', 'red'),
        (conftr_train_losses, conftr_std_train_losses, 'Conftr Train loss', 'blue'),
        title = 'Training Loss per Epoch',
        save_path = os.path.join(results_dir, 'combined_train_losses.png'),
        loss_type = 'Train'
    )

    
    # Plot Set Sizes per epoch
    plot_set_sizes(
        (conftr_set_sizes, conftr_std_set_sizes, 'Conftr Model Set Sizes', 'blue'),
        (base_set_sizes, base_std_set_sizes, 'Baseline Model Set Sizes', 'green'),
        title='Set Sizes per Epoch',
        save_path=os.path.join(results_dir, 'set_sizes_per_epoch.png')
    )

    # Plot accuracies per epoch
    plot_accuracies(
        (conftr_test_accuracies, conftr_std_test_accuracies, 'Conftr Test Accuracy', 'blue'),
        (base_test_accuracies, base_std_test_accuracies, 'Baseline Test Accuracy', 'green'),
        title='Training Accuracy per Epoch',
        save_path=os.path.join(results_dir, 'combined_train_accuracies.png')
    )

    # Evaluate the models on their corresponding test_loader 
    all_results = []
    conftr_results = []
    base_results = []

    for trial in range(num_trials):
        results = eval_main( models_info=[
                            ("conftr_model", config.vr, conftr_trial_params_and_seeds[trial]),
                            ("base_model", config.vr, base_trial_params_and_seeds[trial])
                ])
        all_results.append(results)
        conftr_results.append(results["conftr_model"])
        base_results.append(results["base_model"])


    avg_conftr_results = average_results(conftr_results)
    avg_base_results = average_results(base_results)

    # Add evaluation results to the experiment results dictionary
    experiment_results['evaluation'] = {
        'all_results': all_results,

        'conftr_results': conftr_results,
        'base_results': base_results,
    }
    experiment_results['avg_results'] = {

        'conftr': avg_conftr_results,
        'base': avg_base_results,

    }

    # Plot histogram of average sizes
    avg_sizes = { 
                  "Conftr Model" : avg_conftr_results['avg_size'],
                  "Base Model" : avg_base_results['avg_size'],
           
    }


    plot_avg_sizes(
        *[(name, size) for name, size in avg_sizes.items()],
        title='Average Sizes of Models',
        save_path=os.path.join(results_dir, 'avg_sizes_plot.png')
    )

if __name__ == "__main__":
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


    '''parser = argparse.ArgumentParser(description='Run an experiment or plot results based on the provided configuration.')
    parser.add_argument('action', type=str, choices=['run', 'plot'], help='Action to perform: run experiments or plot results.')
    parser.add_argument('experiment', type=str, choices=['mnist', 'fmnist'], help='The experiment to run or plot.')
    parser.add_argument('subcommand', type=str, nargs='?', choices=['tune', 'tuneplot'], help="Optional subcommand for additional functionality, e.g., 'tune'.")

    args = parser.parse_args()
    # Print parsed arguments
    print(f"Action: {args.action}, Experiment: {args.experiment}, Subcommand: {args.subcommand}")'''
    #run_test_fmnist()
    #resultsdir = os.path.join(os.getcwd(), 'ICLR_sort_tuning_mnist')
    #print(resultsdir)
    #run_tuning(results_dir=resultsdir, config_func = get_experiment_config_fmnist())
    #oldresultsdir = os.path.join(os.getcwd(), 'ICLR_Results/sort_tuning_mnist')
    #plot_tuning(results_dir=resultsdir, old_results_dir = oldresultsdir)
    run_baseline_test("mnist")
    print("Runnign FMNIST NOW")
    run_baseline_test("fmnist")



    
    
    '''print(f"Action: {args.action}, Experiment: {args.experiment}")

    if args.action == 'run':
        if args.experiment == 'mnist':
            if args.subcommand == 'tune':
                resultsdir = os.path.join(os.getcwd(), 'sort_tuning_mnist')
                run_tuning(results_dir=resultsdir, config_func = get_experiment_config_mnist())
            if args.subcommand == 'tuneplot':
                resultsdir = os.path.join(os.getcwd(), 'tune_mnist_ICLR')
                oldresultsdir = os.path.join(os.getcwd(), 'sort_tuning_mnist')
                plot_tuning(results_dir=resultsdir, old_results_dir = oldresultsdir)
            else:
                run_experiment_mnist()
        elif args.experiment == 'fmnist':
            run_experiment_fmnist()'''

            