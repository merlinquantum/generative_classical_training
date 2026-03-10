import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
import random


def pack_params(params_mzi: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    # params_mzi: (M,2), gamma: (m,)
    return jnp.concatenate([params_mzi.reshape(-1), gamma.reshape(-1)], axis=0)


def unpack_params(w: jnp.ndarray, m: int):
    M = m * (m - 1) // 2
    n_mzi = 2 * M
    params_mzi = w[:n_mzi].reshape((M, 2))
    gamma = w[n_mzi:n_mzi + m]
    return params_mzi, gamma


def generate_init_state(m, n, init_state_type = 'beginning'):
    init_state = jnp.zeros(m, dtype=int)

    if init_state_type == 'beginning':
        init_state = init_state.at[:n].set(1)
    elif init_state_type == 'end':
        init_state = init_state.at[-n:].set(1)
    elif init_state_type == 'beginning_alternating':
        init_state = init_state.at[:2*n-1:2].set(1)
    elif init_state_type == 'middle_compact':
        start = (m - n) // 2
        init_state = init_state.at[start:start+n].set(1)
    elif init_state_type == 'middle_alternating':
        span = 2*n - 1
        start = (m - span) // 2
        init_state = init_state.at[start : start + span : 2].set(1)
    else:
        print('Type not supported')

    return init_state


def random_bitstrings(m, n, num_samples):
    """
    Uniformly generate num_samples binary strings of length m with Hamming weight n
    :param m: length of bitstrings
    :param n: Hamming weight
    :param num_samples: number of samples to create
    """
    bitstrings = []
    for _ in range(num_samples):
        bits = [1] * n + [0] * (m - n)
        random.shuffle(bits)
        bitstrings.append(bits)
    
    return np.array(bitstrings)


def sample_bitstrings(key, m, p, num_samples):
    """
    Sample num_samples binary strings of length m by independent Bernoulli(p) trials.
    :param key: PRNGKey
    :param m: length of bitstrings
    :param p: probability of success in Bernoulli distribution
    :param num_samples: number of strings to create
    :return: array of shape (num_samples, n) with dtype int32
    """
    # Draw uniform randoms in [0,1)
    u = jax.random.uniform(key, shape=(num_samples, m))
    # Compare to p and cast to int
    return (u < p).astype(jnp.int32)


def p_sigma(sigma):
    return (1-jnp.exp(-1/(2*sigma**2)))/2


# Adapted from: https://github.com/XanaduAI/iqpopt
# License: Apache License 2.0
def median_heuristic(X):
    """
    Compute an estimate of the median heuristic used to decide the bandwidth of the RBF kernels; see
    https://arxiv.org/abs/1707.07269
    :param X (array): Dataset of interest
    :return (float): median heuristic estimate
    """
    m = len(X)
    X = np.array(X)
    med = np.median([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for i in range(m) for j in range(m)])
    return med


# Adapted from: https://github.com/XanaduAI/iqpopt
# License: Apache License 2.0
def gaussian_kernel(sigma: float, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Calculates the value for the gaussian kernel between two vectors x, y

    Args:
        sigma (float): sigma parameter, the width of the kernel
        x (jnp.ndarray): one of the vectors
        y (jnp.ndarray): the other vector

    Returns:
        float: Result value of the gaussian kernel
    """
    return jnp.exp(-((x-y)**2).sum()/2/sigma**2)
