import ml_collections as collections
import jax.numpy as jnp
from models import * 
import os
################## MNIST ################
def get_conformal_config_mnist() -> collections.ConfigDict:
    "Configurations for conftr mnist centralized"
    config = collections.ConfigDict()


    #dataset config
    config.dataset_name = 'mnist'
    config.train_size = 55000
    config.calib_size = 5000
    config.test_size = 10000
    config.batch_size = 500

    #model config
    config.model_class = Linear_mnist
    config.batch_input_shape = (config.batch_size, 1, 28, 28)
    config.num_labels = 10
    config.num_inputs = jnp.prod(jnp.array(config.batch_input_shape[1:]))


    #conformal training loss hyperparams
    config.temperature = 0.5
    config.target_size = 1
    config.confidence_threshold = 0.01
    config.alpha = config.confidence_threshold
    config.regularizer_weight = 0.0005
    config.base_loss_weight = 0
    config.coverage_weight = 0 
    config.size_weight = 0.01
    config.loss_matrix = jnp.eye(config.num_labels)


    #training configs
    config.epochs = 50

    #lr scheduler configs
    config.lr_scheduler = collections.ConfigDict()
    config.lr_scheduler.learning_rate = 0.05
    config.lr_scheduler.learning_rate_decay = 0.1

    #optimizer setting
    config.optimizer_decay = 0.9

    #Sort Number
    config.num_sort = 6


    return config

def get_experiment_config_mnist():
    config = collections.ConfigDict()

   
    config.vr = get_conformal_config_mnist
   

    config.num_trials = 1

    #results dir
    config.results_dir = os.path.join(os.getcwd(), 'MNIST_RESULTS')
    #model paths
    config.conftr_model_path = 'conftr_model_mnist.pkl'
    config.base_model_path = 'base_model_mnist.pkl'
    config.sort_model_path = 'sort_model_mnist.pkl'
    config.split_model_path = 'split_model_mnist.pkl'
    #loss plot settings
    config.conftr_train_loss_plot = {
        'title': 'ConfTR Training Loss - MNIST',
        'filename': 'conftr_train_loss_mnist.png'
    }
    config.conftr_test_loss_plot = {
        'title': 'ConfTR Test Loss - MNIST',
        'filename': 'conftr_test_loss_mnist.png'
    }
    config.sort_train_loss_plot = {
        'title': 'sort Training Loss - MNIST',
        'filename': 'sort_train_loss_mnist.png'
    }
    config.sort_test_loss_plot = {
        'title': 'sort Test Loss - MNIST',
        'filename': 'sort_test_loss_mnist.png'
    }
    config.base_train_loss_plot = {
        'title': 'Baseline Training Loss - MNIST',
        'filename': 'base_train_loss_mnist.png'
    }
    config.base_test_loss_plot = {
        'title': 'Baseline Test Loss - MNIST',
        'filename': 'base_test_loss_mnist.png'
    }

    return config

############  For FMNIST ###################
def get_conformal_config_fmnist() -> collections.ConfigDict:
    "Configurations for conftr mnist centralized"
    config = collections.ConfigDict()

    #dataset config
    config.dataset_name = 'fmnist'
    config.train_size = 55000
    config.calib_size = 5000
    config.test_size = 10000
    config.batch_size = 500

    #model config
    config.batch_input_shape = (config.batch_size, 1, 28, 28)
    config.num_labels = 10
    config.num_inputs = jnp.prod(jnp.array(config.batch_input_shape[1:]))


    #conformal training loss hyperparams
    config.temperature = 0.1
    config.target_size = 0
    config.confidence_threshold = 0.01
    config.alpha = config.confidence_threshold
    config.regularizer_weight = 0.0005
    config.base_loss_weight = 0
    config.coverage_weight = 0 
    config.size_weight = 0.01
    config.loss_matrix = jnp.eye(config.num_labels)


    #training configs
    config.epochs = 100
    

    #lr scheduler configs
    config.lr_scheduler = collections.ConfigDict()
    config.lr_scheduler.learning_rate = 0.01
    config.lr_scheduler.learning_rate_decay = 0.1

    #optimizer setting
    config.optimizer_decay = 0.9

    #sort number
    config.num_sort = 4
    
    
    return config

def conftr_100() -> collections.ConfigDict:
    "Configurations for conftr mnist centralized"
    config = collections.ConfigDict()

    #dataset config
    config.dataset_name = 'fmnist'
    config.train_size = 55000
    config.calib_size = 5000
    config.test_size = 10000
    config.batch_size = 100

    #model config
    config.batch_input_shape = (config.batch_size, 1, 28, 28)
    config.num_labels = 10
    config.num_inputs = jnp.prod(jnp.array(config.batch_input_shape[1:]))


    #conformal training loss hyperparams
    config.temperature = 0.1
    config.target_size = 0
    config.confidence_threshold = 0.01
    config.alpha = config.confidence_threshold
    config.regularizer_weight = 0.0005
    config.base_loss_weight = 0
    config.coverage_weight = 0 
    config.size_weight = 0.01
    config.loss_matrix = jnp.eye(config.num_labels)


    #training configs
    config.epochs = 100
    

    #lr scheduler configs
    config.lr_scheduler = collections.ConfigDict()
    config.lr_scheduler.learning_rate = 0.01
    config.lr_scheduler.learning_rate_decay = 0.1

    #optimizer setting
    config.optimizer_decay = 0.9

    #sort number
    config.num_sort = 4
    
    
    return config
def get_experiment_config_fmnist():
    config = collections.ConfigDict()

    config.vr = get_conformal_config_fmnist
    config.ct100 = conftr_100


    config.num_trials = 1

    #results dir
    config.results_dir = os.path.join(os.getcwd(), 'Fashion-MNIST_RESULTS')
    #model paths
    config.conftr_model_path = 'conftr_model_fmnist.pkl'
    config.base_model_path = 'base_model_fmnist.pkl'
    config.sort_model_path = 'sort_model_fmnist.pkl'
    config.split_model_path = 'split_model_fmnist.pkl'
    #loss plot settings
    config.conftr_train_loss_plot = {
        'title': 'ConfTR Training Loss - F-MNIST',
        'filename': 'conftr_train_loss_fmnist.png'
    }
    config.conftr_test_loss_plot = {
        'title': 'ConfTR Test Loss - F-MNIST',
        'filename': 'conftr_test_loss_fmnist.png'
    }
    config.sort_train_loss_plot = {
        'title': 'sort Training Loss - F-MNIST',
        'filename': 'sort_train_loss_fmnist.png'
    }
    config.sort_test_loss_plot = {
        'title': 'sort Test Loss - F-MNIST',
        'filename': 'sort_test_loss_fmnist.png'
    }
    config.base_train_loss_plot = {
        'title': 'Baseline Training Loss - F-MNIST',
        'filename': 'base_train_loss_fmnist.png'
    }
    config.base_test_loss_plot = {
        'title': 'Baseline Test Loss -F-MNIST',
        'filename': 'base_test_loss_fmnist.png'
    }

    return config
