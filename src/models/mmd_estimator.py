import jax.numpy as jnp
import jax
from functools import partial
from ..helpers.utils import p_sigma, sample_bitstrings, unpack_params
from ..helpers.gurvits import sample_Z_dataset, glynn_on_dataset
from ..helpers.circuit import clements_unitary, butterfly_unitary, mzi3_unitary, haar_unitary

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=["n"])
def compute_single_op(k, U, n, Z, target_dataset, init_state_ind):
    """
    Subfunction of MMD loss computation
    Get value for single Z_k operator
    """
    # Get Pauli Zk operator and correspond LO matrix whose permanent we wish to compute
    Z_k = jnp.diag(1 - 2*k).astype(jnp.complex64)
    A = (jnp.conj(U.T) @ Z_k @ U)[init_state_ind[:, None], init_state_ind]

    # Get estimate on LO circuit, obtain array for all samples from Z set
    G_samples = glynn_on_dataset(A, Z)
    Z_len = len(Z)
    G = jnp.mean(G_samples)
    G_corr = jnp.mean(G_samples**2)/Z_len

    # Get estimate on target data
    T = jnp.mean(1-2*((target_dataset @ k.T) % 2), axis=0)
    T_len = len(target_dataset)

    # Get unbiased MMD for one Pauli operator
    MMD_single_op = (G*G-G_corr)*Z_len/(Z_len-1) - 2*G*T + (T*T*T_len-1)/(T_len-1)

    return MMD_single_op


@partial(jax.jit, static_argnames=["m", "n", "n_samples_operators", "n_samples_gurvits"])
def MMD_loss(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators,
             n_samples_gurvits, init_state_ind):
    """
    A function to estimate the value of the MMD^2 given linear optical circuit parameters and target dataset
    Adapted for Clements decomposition

    :param circuit_parameters: parameters of the LO circuit
    :param target_dataset: target dataset for learning problem
    :param sigma: bandwidth for the MMD, can be a single value or a list of values
    :param m: number of modes
    :param n: number of photons
    :param key: PRNGKey
    :param n_samples_operators: number of samples for operator terms in MMD observable
    :param n_samples_gurvits: number of samples for Gurvits algorithm
    :return: value of the MMD^2 loss
    """
    sigma = jnp.asarray(sigma)
    sigmas = [sigma] if sigma.ndim == 0 else list(sigma)

    losses = []
    for sigma in sigmas:
        # Samples operators for MMD 
        p_MMD = p_sigma(sigma)
        key2, k_key = jax.random.split(key, 2)
        K = jnp.array(jax.random.binomial(k_key, 1, p_MMD, shape=(n_samples_operators, m)), dtype='float64')
    
        # Get LO unitary corresponding to parameters
        params_mzi, gammas = unpack_params(circuit_parameters, m)
        U = clements_unitary(params_mzi, gammas, m)
    
        # Sample bitstring dataset for Gurvits algorithm
        key3, z_key = jax.random.split(key2, 2)
        Z = sample_Z_dataset(z_key, n, n_samples_gurvits)
        
        compute_all_ops = jax.vmap(
            lambda k: compute_single_op(k, U, n, Z, target_dataset, init_state_ind), in_axes=0)
    
        losses.append(jnp.real(jnp.mean(compute_all_ops(K))))

    return sum(losses)/len(losses)


@partial(jax.jit, static_argnames=["m", "n", "n_samples_operators", "n_samples_gurvits"])
def MMD_loss_butterfly(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators,
             n_samples_gurvits, init_state_ind):
    """
    A function to estimate the value of the MMD^2 given linear optical circuit parameters and target dataset
    Adapted for Butterfly ansatz

    :param circuit_parameters: parameters of the LO circuit
    :param target_dataset: target dataset for learning problem
    :param sigma: bandwidth for the MMD
    :param m: number of modes
    :param n: number of photons
    :param key: PRNGKey
    :param n_samples_operators: number of samples for operator terms in MMD observable
    :param n_samples_gurvits: number of samples for Gurvits algorithm
    :return: value of the MMD^2 loss
    """
    sigma = jnp.asarray(sigma)
    sigmas = [sigma] if sigma.ndim == 0 else list(sigma)

    losses = []
    for sigma in sigmas:
        # Samples operators for MMD
        p_MMD = p_sigma(sigma)
        key2, k_key = jax.random.split(key, 2)
        K = jnp.array(jax.random.binomial(k_key, 1, p_MMD, shape=(n_samples_operators, m)), dtype='float64')
    
        # Get LO unitary corresponding to parameters
        params_mzi, gammas = unpack_params(circuit_parameters, m)
        U = butterfly_unitary(params_mzi, gammas, m)
    
        # Sample bitstring dataset for Gurvits algorithm
        key3, z_key = jax.random.split(key2, 2)
        Z = sample_Z_dataset(z_key, n, n_samples_gurvits)
    
        compute_all_ops = jax.vmap(
            lambda k: compute_single_op(k, U, n, Z, target_dataset, init_state_ind), in_axes=0)

        losses.append(jnp.real(jnp.mean(compute_all_ops(K))))

    return sum(losses)/len(losses)


@partial(jax.jit, static_argnames=["m", "n", "n_samples_operators", "n_samples_gurvits"])
def MMD_loss_mzi3(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators,
             n_samples_gurvits, init_state_ind):
    """
    A function to estimate the value of the MMD^2 given linear optical circuit parameters and target dataset
    Adapted for MZI3 ansatz

    :param circuit_parameters: parameters of the LO circuit
    :param target_dataset: target dataset for learning problem
    :param sigma: bandwidth for the MMD, can be a single value or a list of values
    :param m: number of modes
    :param n: number of photons
    :param key: PRNGKey
    :param n_samples_operators: number of samples for operator terms in MMD observable
    :param n_samples_gurvits: number of samples for Gurvits algorithm
    :return: value of the MMD^2 loss
    """
    sigma = jnp.asarray(sigma)
    sigmas = [sigma] if sigma.ndim == 0 else list(sigma)

    losses = []
    for sigma in sigmas:
        # Samples operators for MMD
        p_MMD = p_sigma(sigma)
        key2, k_key = jax.random.split(key, 2)
        K = jnp.array(jax.random.binomial(k_key, 1, p_MMD, shape=(n_samples_operators, m)), dtype='float64')
    
        # Get LO unitary corresponding to parameters
        params_mzi, gammas = unpack_params(circuit_parameters, m)
        U = mzi3_unitary(params_mzi, gammas, m)
    
        # Sample bitstring dataset for Gurvits algorithm
        key3, z_key = jax.random.split(key2, 2)
        Z = sample_Z_dataset(z_key, n, n_samples_gurvits)
    
        compute_all_ops = jax.vmap(
            lambda k: compute_single_op(k, U, n, Z, target_dataset, init_state_ind), in_axes=0)
    
        losses.append(jnp.real(jnp.mean(compute_all_ops(K))))

    return sum(losses)/len(losses)


@partial(jax.jit, static_argnames=["m", "n", "n_samples_operators", "n_samples_gurvits"])
def MMD_loss_haar(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators,
                  n_samples_gurvits, init_state_ind):
    """
    A function to estimate the value of the MMD^2 given linear optical circuit parameters and target dataset
    Adapted for Haar compatible ansatz

    :param circuit_parameters: parameters of the LO circuit
    :param target_dataset: target dataset for learning problem
    :param sigma: bandwidth for the MMD, can be a single value or a list of values
    :param m: number of modes
    :param n: number of photons
    :param key: PRNGKey
    :param n_samples_operators: number of samples for operator terms in MMD observable
    :param n_samples_gurvits: number of samples for Gurvits algorithm
    :return: value of the MMD^2 loss
    """
    sigma = jnp.asarray(sigma)
    sigmas = [sigma] if sigma.ndim == 0 else list(sigma)

    losses = []
    for sigma in sigmas:
        # Samples operators for MMD 
        p_MMD = p_sigma(sigma)
        key2, k_key = jax.random.split(key, 2)
        K = jnp.array(jax.random.binomial(k_key, 1, p_MMD, shape=(n_samples_operators, m)), dtype='float64')
    
        # Get LO unitary corresponding to parameters
        U = haar_unitary(circuit_parameters, m)
    
        # Sample bitstring dataset for Gurvits algorithm
        key3, z_key = jax.random.split(key2, 2)
        Z = sample_Z_dataset(z_key, n, n_samples_gurvits)
        
        compute_all_ops = jax.vmap(
            lambda k: compute_single_op(k, U, n, Z, target_dataset, init_state_ind), in_axes=0)
    
        losses.append(jnp.real(jnp.mean(compute_all_ops(K))))

    return sum(losses)/len(losses)
  
    