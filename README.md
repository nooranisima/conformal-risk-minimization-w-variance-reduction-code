
# Confromal Risk Minimization with Variance Reduction
This is the official repository for our paper ["Conformal Risk Minimization with Variance Reduction."](#) For more information, please refer to our [project website](https://nooranisima.github.io/conformal-risk-minimization-with-variance-reduction/).

This repository implements Conftr and Vr-Conftr, with a sorting-based estimator for the gradient of the population quantile, and conducts experiments for training models.

In regard to the `requirements.txt`, ensure you have the `jax`, `flax`, `flaxmodels`, `matplotlib`, and `ml_collections` libraries installed before running the code. Alternatively, you can set up the conda environment by running:

```bash
conda env create -f environment.yml
conda activate vr_env
```
## Table of Contents

- [Repository Structure](#repository-structure)
  - [algorithms/](#algorithms)
  - [config/](#config)
  - [data/](#data)
  - [models/](#models)
  - [experiments/](#experiments)
  - [evaluation/](#evaluation)
  - [environment.yml](#environment-yml)
- [Designing a New Experiment](#designing-a-new-experiment)
- [Reproducing Existing Experiments](#reproducing-existing-experiments)

## Repository Structure

### algorithms/

This folder contains the core algorithms used in this repository, including an implementation of the ConfTr method, and Vr-ConfTr with a sorting-based estimator.

- `__init__.py`: Initializes the algorithms module.
- `conftr.py`: Implements the ConfTr (Conformal Training) method.
- `vr_sort.py`: Implements the VR-ConfTr method with sorting-based estimator for the gradient of the population quantile
### config/

This folder contains configuration files for setting up experiments and models.

- `__init__.py`: Initializes the config module.
- `config.py`: Contains configuration settings for each algorithm, as well as various experiments, including hyperparameters, model settings, and dataset paths.

### data/

This folder contains scripts related to data handling and preprocessing. It currently supports MNIST, and FashionMNIST with specific preprocessing and augmentation steps applied as described in the paper. This script can be easily modified to include more datasets. 

- `__init__.py`: Initializes the data module.
- `data_loader.py`: This script handles data loading and preprocessing for various datasets.
  - **Supported Datasets:** Defines supported datasets: MNIST, FashionMNIST
  - **Data Loaders:** Contains functions to load and split datasets into training, (calibration), and test sets.
  - **Random Seed Handling:** Ensures reproducibility by setting random seeds for PyTorch, NumPy, and Jax.

### models/

This folder contains model definitions used in the experiments.
- `__init__.py`: Initializes the models module.
- `models.py`: Contains the implementation of the models used in the experiments. Alternate architectures can be seamlessly integrated in the code. 

### experiments/

This folder contains scripts to run various experiments involving different models, datasets, and algorithms. These experiments are for comparing the performance of ConfTr and VR-ConfTr
- `__init__.py`: Initializes the experiments module.
- `experiment.py`: This script is the main driver for running experiments in the repository. It supports a range of configurations and allows for the training, evaluation, and fine-tuning of models.
  - **Evaluation:** The script evaluates models using both accuracy and conformal prediction metrics. It also plots combined training and test losses for each algorithm.
  - **Result Compilation:** The script aggregates results from multiple trials and generates plots for the training trajectories of each algorithm. These results are saved in a specified results directory for further analysis.
  - **Main Experiment Functions:**
    - `run_experiment_mnist`: Runs training on the MNIST dataset.
    - `run_experiment_fmnist`: Runs training on the Fashion MNIST dataset.

### evaluation/

This folder contains scripts for evaluating the performance of the models trained using various algorithms. The evaluation includes accuracy assessment and conformal prediction metrics, such as coverage rates and prediction set sizes. The evaluation process is essential for comparing the effectiveness of different training methods and models.

- `__init__.py`: Initializes the evaluation module.
- `evaluation.py`: This script provides a comprehensive evaluation framework for models. It includes the following key components:
  - **Model Accuracy Evaluation:** The `compute_accuracy` function calculates the accuracy of a given model on a test dataset.
  - **Conformal Prediction Evaluation:** The `evaluate_conformal_prediction` function performs split conformal prediction to assess coverage rates and prediction set sizes. It shuffles and splits calibration and test datasets, computes non-conformity scores, and evaluates the prediction sets against specified alpha thresholds.
  - **Main Evaluation Loop:** The `main` function allows users to load models, configure datasets, and perform evaluations. It returns a dictionary containing the evaluation results for each model, which includes accuracy, average coverage rates, standard deviation of coverage rates, average sizes, and standard deviation of sizes.

### environment.yml

This file contains the environment configuration for setting up the required dependencies using conda.

## Designing a New Experiment

To design a new experiment, follow these steps:

1. **Set up the configuration:** Create a new configuration file in the `config/` folder.
2. **Define the model:** If your experiment uses a new model, define it in the `models/` folder.
3. **Implement the algorithm:** If your experiment involves a new algorithm, add it to the `algorithms/` folder.
4. **Prepare the data:** Ensure the data loader in the `data/` folder is set up to handle your dataset.
5. **Run the experiment:** Use the `experiment.py` script in the `experiments/` folder to run your experiment.
  
The above architectures are set-up to easily handle multiple training and testing trials. The results averaged over all the runs will be saved in the current directory. All generated figures will be saved, as well as the explicit trajectories so that they can be replotted afterwards if needed. 
## Reproducing Existing Experiments

To reproduce the existing experiments, follow these steps:

1. **Set up the environment:** Use the `requirements.txt` file to install the required dependencies. 
2. **Choose an experiment:** Run experiment.py and specify the dataset from ['mnist', 'fmnist'] to run the correpsonding training.
