# import jax.numpy as jnp
from .sorting_nets import comm_pattern_batcher
from .variational_sorting_net import VariationalSortingNet


# def smooth_quantile(x, p):
#     return jnp.quantile(x, p)  # For now, jax's built-in sample quantile function will do

def smooth_quantile(array, prob, dispersion=0.1):
    comm = comm_pattern_batcher(len(array), make_parallel=True)
    
    sos = VariationalSortingNet(comm,
                                smoothing_strategy='entropy_reg',
                                sorting_strategy='hard')   
    
    return sos.quantile(array,
                        dispersion=dispersion,
                        alpha=prob,
                        tau=0.5
                        )
