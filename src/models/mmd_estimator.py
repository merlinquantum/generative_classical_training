import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional

from ..helpers.utils import sample_bitstrings, unpack_params
from ..helpers.kernels import p_sigma, sample_kernel_operators
from ..helpers.gurvits import (
    sample_Z_dataset,
    glynn_on_dataset,
    generalized_glynn_on_dataset,
    unbiased_product_estimate,
    sample_roots_dataset,
)
from ..helpers.circuit import (
    clements_unitary,
    butterfly_unitary,
    mzi3_unitary,
    haar_unitary,
)

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray




# ************************ HELPERS **************************************************

def unitary_from_ansatz(circuit_parameters: Array, m: int, ansatz: str) -> Array:
    """
    Build the unitary associated with the selected ansatz, such that: 

    For mesh ansatzes, circuit_parameters is the packed vector containing MZI parameters and output phases. 
    For Haar, circuit_parameters is passed directly to haar_unitary.
    """
    if ansatz == "haar":
        return haar_unitary(circuit_parameters, m)

    params_mzi, gammas = unpack_params(circuit_parameters, m)

    if ansatz == "clements":
        return clements_unitary(params_mzi, gammas, m)
    if ansatz == "butterfly":
        return butterfly_unitary(params_mzi, gammas, m)
    if ansatz == "mzi3":
        return mzi3_unitary(params_mzi, gammas, m)

    raise ValueError(f"Unknown ansatz: {ansatz}")


def random_bitstrings_fixed_weight(num_samples: int, m: int, n: int, seed: int = 0) -> Array:
    """ Generate random bitstrings of length m with fixed Hamming weight n"""
    rng = np.random.default_rng(seed)
    scores = rng.random((num_samples, m))
    inds = np.argpartition(scores, kth=n - 1, axis=1)[:, :n]
    X = np.zeros((num_samples, m), dtype=np.float64)
    rows = np.arange(num_samples)[:, None]
    X[rows, inds] = 1.0
    return jnp.asarray(X)


@jax.jit
def dataset_expectations(K: Array, dataset: Array) -> Array:
    """ Compute E_x[(-1)^(k.x)] for every row k in K :
    - K has shape (n_operators, m).
    - dataset has shape (n_samples, m).
    --> Returns shape (n_operators,).
    """
    parity = (dataset @ K.T) % 2.0
    values = 1.0 - 2.0 * parity
    return jnp.mean(values, axis=0)


@partial(jax.jit, static_argnames=("n",))
def compute_single_op(k: Array, U: Array, n: int, Z: Array, target_dataset: Array, init_state_ind: Array) -> Array:
    """
    Compute the unbiased MMD contribution of one Walsh operator.
    --> Optimization: instead of building diag(1 - 2k), we multiply the rows of U
    directly by the diagonal entries. This avoids allocating an m x m diagonal matrix for every sampled operator.
    """
    z_diag = (1.0 - 2.0 * k).astype(U.dtype)
    weighted_U = z_diag[:, None] * U
    A_full = jnp.conj(U.T) @ weighted_U
    A = A_full[init_state_ind[:, None], init_state_ind]

    G_samples = glynn_on_dataset(A, Z)
    Z_len = Z.shape[0]
    G = jnp.mean(G_samples)
    G_corr = jnp.mean(G_samples**2) / Z_len

    T = jnp.mean(1.0 - 2.0 * ((target_dataset @ k.T) % 2.0), axis=0)
    T_len = target_dataset.shape[0]

    return (G * G - G_corr) * Z_len / (Z_len - 1) - 2.0 * G * T + (T * T * T_len - 1.0) / (T_len - 1)


@partial(jax.jit, static_argnames=("n",))
def compute_all_ops(K: Array, U: Array, n: int, Z: Array, target_dataset: Array, init_state_ind: Array) -> Array:
    return jax.vmap(lambda k: compute_single_op(k, U, n, Z, target_dataset, init_state_ind))(K)





# MMD LOSSES 

@partial(jax.jit,static_argnames=("m","n","n_samples_operators","n_samples_gurvits","ansatz","kernel_type","degree","max_weight"))
def MMD_loss_general(
    circuit_parameters: Array,
    target_dataset: Array,
    sigma: float,
    m: int,
    n: int,
    key: Array,
    n_samples_operators: int,
    n_samples_gurvits: int,
    init_state_ind: Array,
    mode_probs: Optional[Array] = None,
    ansatz: str = "clements",
    kernel_type: str = "gaussian",
    degree: int = 2,
    c: float = 1.0,
    max_weight: int = 2,
) -> Array:
    
    """Generic MMD^2 estimator for Haar, Clements, Butterfly and MZI3"""
    key_k, key_z = jax.random.split(key, 2)

    K = sample_kernel_operators(
        key=key_k,
        sigma=sigma,
        mode_probs=mode_probs,
        m=m,
        n=n,
        n_samples_operators=n_samples_operators,
        kernel_type=kernel_type,
        degree=degree,
        c=c,
        max_weight=max_weight,
    )

    U = unitary_from_ansatz(circuit_parameters, m, ansatz)
    Z = sample_Z_dataset(key_z, n, n_samples_gurvits)

    mmd_terms = compute_all_ops(K, U, n, Z, target_dataset, init_state_ind)
    return jnp.real(jnp.mean(mmd_terms))


# Backward-compatible wrappers
@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits"))
def MMD_loss(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators, n_samples_gurvits, init_state_ind):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs=None,
        ansatz="clements",
        kernel_type="gaussian",
    )

MMD_loss_compact = MMD_loss

@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits"))
def MMD_loss_butterfly(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators, n_samples_gurvits, init_state_ind):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs=None,
        ansatz="butterfly",
        kernel_type="gaussian",
    )


@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits"))
def MMD_loss_mzi3(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators, n_samples_gurvits, init_state_ind):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs=None,
        ansatz="mzi3",
        kernel_type="gaussian",
    )


@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits"))
def MMD_loss_haar(circuit_parameters, target_dataset, sigma, m, n, key, n_samples_operators, n_samples_gurvits, init_state_ind):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs=None,
        ansatz="haar",
        kernel_type="gaussian",
    )


@partial(jax.jit,static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits", "kernel_type", "degree", "max_weight"))
def MMD_loss_haar_general(
    params,
    target_dataset,
    sigma,
    m,
    n,
    key,
    n_samples_operators,
    n_samples_gurvits,
    init_state_ind,
    mode_probs,
    kernel_type="gaussian",
    degree=2,
    c=1.0,
    max_weight=2,
):
    return MMD_loss_general(
        params,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs,
        ansatz="haar",
        kernel_type=kernel_type,
        degree=degree,
        c=c,
        max_weight=max_weight,
    )


@partial(jax.jit,static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits", "kernel_type", "degree", "max_weight"))
def MMD_loss_butterfly_general(
    circuit_parameters,
    target_dataset,
    sigma,
    m,
    n,
    key,
    n_samples_operators,
    n_samples_gurvits,
    init_state_ind,
    mode_probs,
    kernel_type="gaussian",
    degree=2,
    c=1.0,
    max_weight=2,
):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs,
        ansatz="butterfly",
        kernel_type=kernel_type,
        degree=degree,
        c=c,
        max_weight=max_weight,
    )


@partial(jax.jit,static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits", "kernel_type", "degree", "max_weight"))
def MMD_loss_clements_general(
    circuit_parameters,
    target_dataset,
    sigma,
    m,
    n,
    key,
    n_samples_operators,
    n_samples_gurvits,
    init_state_ind,
    mode_probs,
    kernel_type="gaussian",
    degree=2,
    c=1.0,
    max_weight=2,
):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs,
        ansatz="clements",
        kernel_type=kernel_type,
        degree=degree,
        c=c,
        max_weight=max_weight,
    )


@partial(jax.jit,static_argnames=("m", "n", "n_samples_operators", "n_samples_gurvits", "kernel_type", "degree", "max_weight"))
def MMD_loss_mzi3_general(
    circuit_parameters,
    target_dataset,
    sigma,
    m,
    n,
    key,
    n_samples_operators,
    n_samples_gurvits,
    init_state_ind,
    mode_probs,
    kernel_type="gaussian",
    degree=2,
    c=1.0,
    max_weight=2,
):
    return MMD_loss_general(
        circuit_parameters,
        target_dataset,
        sigma,
        m,
        n,
        key,
        n_samples_operators,
        n_samples_gurvits,
        init_state_ind,
        mode_probs,
        ansatz="mzi3",
        kernel_type=kernel_type,
        degree=degree,
        c=c,
        max_weight=max_weight,
    )


