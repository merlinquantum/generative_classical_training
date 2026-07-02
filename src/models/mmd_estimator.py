import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional

from ..helpers.utils import p_sigma, sample_bitstrings, unpack_params
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



# ************************************ Kernel Probabilities ********************************************************
def p_laplacian(lam: float) -> Array:
    """Bernoulli parameter for the laplacian kernel"""
    return (1.0 - jnp.exp(-lam)) / 2.0


def compute_frequency_weights(mode_probs: Array, eps: float = 1e-8) -> Array:
    """Data-adaptive weights normalized to mean 1."""
    raw_weights = jnp.asarray(mode_probs, dtype=jnp.float64) + eps
    return raw_weights / jnp.mean(raw_weights)


def p_weighted_gaussian(sigma: float, mode_probs: Array, eps: float = 1e-8) -> Array:
    """Return Bernouilli parameters for the weighted Gaussian Kernel"""
    weights = compute_frequency_weights(mode_probs, eps=eps)
    alpha = weights / (2.0 * sigma**2)
    return (1.0 - jnp.exp(-alpha)) / 2.0



# ******************************** Kernel_Sampling ******************************************************
@partial(jax.jit, static_argnames=("m", "n_samples_operators"))
def sample_bernoulli_kernel_operators(key: Array, p: Array, m: int, n_samples_operators: int) -> Array:
    return jax.random.bernoulli(key, p=p, shape=(n_samples_operators, m)).astype(jnp.float64)


@partial(jax.jit, static_argnames=("m", "n_samples_operators", "degree"))
def sample_parity_polynomial_operators(key: Array,m: int,n_samples_operators: int,degree: int,p_feature_total: float) -> Array:
    """ Generic sampler for polynomial-type Walsh operators.
    --> Each factor either chooses the constant term or one parity feature.
    Repeated feature indices cancel modulo 2.
    """
    def sample_one(subkey):
        keys = jax.random.split(subkey, degree)

        def body(k_current, key_step):
            key_choice, key_index = jax.random.split(key_step)
            choose_feature = jax.random.bernoulli(key_choice, p=p_feature_total)
            idx = jax.random.randint(key_index, shape=(), minval=0, maxval=m)

            return jax.lax.cond(
                choose_feature,
                lambda kk: kk.at[idx].set(1.0 - kk[idx]),
                lambda kk: kk,
                k_current,
            ), None

        k0 = jnp.zeros(m, dtype=jnp.float64)
        k_final, _ = jax.lax.scan(body, k0, keys)
        return k_final

    keys = jax.random.split(key, n_samples_operators)
    return jax.vmap(sample_one)(keys)


@partial(jax.jit, static_argnames=("m", "n_samples_operators", "degree"))
def sample_polynomial_kernel_operators(key: Array,m: int,n_samples_operators: int,degree: int = 2,c: float = 1.0) -> Array:
    p_feature_total = 1.0 / (c + 1.0)
    return sample_parity_polynomial_operators(key=key,m=m,n_samples_operators=n_samples_operators,degree=degree,p_feature_total=p_feature_total)


@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "degree"))
def sample_polynomial_1_kernel_operators(key: Array,m: int,n: int,n_samples_operators: int,degree: int = 2,c: float = 20.0) -> Array:
    """Sampler for the standard polynomial kernel on fixed-weight data."""
    a = c + n - m / 4.0
    b = 1.0 / 4.0
    normalizer = c + n
    p_feature_total = (m * b) / normalizer

    # If a < 0, the sampling interpretation is not valid, we check : 
    _ = a

    return sample_parity_polynomial_operators(key=key,m=m,n_samples_operators=n_samples_operators,degree=degree,p_feature_total=p_feature_total)


@partial(jax.jit, static_argnames=("m", "n_samples_operators"))
def sample_weighted_gaussian_kernel_operators(key: Array,sigma: float,mode_probs: Array,m: int,n_samples_operators: int) -> Array:
    p = p_weighted_gaussian(sigma=sigma, mode_probs=mode_probs)
    return sample_bernoulli_kernel_operators(key, p, m, n_samples_operators)


@partial(jax.jit, static_argnames=("m", "n_samples_operators", "max_weight"))
def sample_low_order_kernel_operators(key: Array,m: int,n_samples_operators: int,max_weight: int = 2) -> Array:
    """ Samples k with |k| in {1, ..., max_weight}, uniformly over modes"""

    def sample_one(subkey):
        key_w, key_perm = jax.random.split(subkey)
        w = jax.random.randint(key_w, shape=(), minval=1, maxval=max_weight + 1)
        perm = jax.random.permutation(key_perm, m)
        selected = perm[:max_weight]
        mask = (jnp.arange(max_weight) < w).astype(jnp.float64)
        k = jnp.zeros(m, dtype=jnp.float64)
        return k.at[selected].set(mask)

    keys = jax.random.split(key, n_samples_operators)
    return jax.vmap(sample_one)(keys)


@partial(jax.jit, static_argnames=("m", "n_samples_operators", "max_weight"))
def sample_data_biased_low_order_kernel_operators(key: Array,mode_probs: Array,m: int,n_samples_operators: int,max_weight: int = 2) -> Array:
    """Samples low-order k vectors with modes chosen according to mode_probs."""
    probs = jnp.asarray(mode_probs, dtype=jnp.float64)
    probs = probs / jnp.sum(probs)

    def sample_one(subkey):
        key_w, key_modes = jax.random.split(subkey)
        w = jax.random.randint(key_w, shape=(), minval=1, maxval=max_weight + 1)
        inds = jax.random.choice(key_modes, a=m, shape=(max_weight,), replace=False, p=probs)
        mask = (jnp.arange(max_weight) < w).astype(jnp.float64)
        k = jnp.zeros(m, dtype=jnp.float64)
        return k.at[inds].set(mask)

    keys = jax.random.split(key, n_samples_operators)
    return jax.vmap(sample_one)(keys)


@partial(jax.jit,static_argnames=("m", "n", "n_samples_operators", "kernel_type", "degree", "max_weight"))
def sample_kernel_operators(key: Array,sigma: float,mode_probs: Optional[Array],m: int,n: int,n_samples_operators: int,kernel_type: str = "gaussian",degree: int = 2,c: float = 1.0,max_weight: int = 2) -> Array:
    """Generic sampler for Walsh operators k according to the chosen kernel """
    if kernel_type == "gaussian":
        return sample_bernoulli_kernel_operators(key, p_sigma(sigma), m, n_samples_operators)

    if kernel_type == "weighted_gaussian":
        if mode_probs is None:
            raise ValueError("mode_probs must be provided for weighted_gaussian.")
        return sample_weighted_gaussian_kernel_operators(key, sigma, mode_probs, m, n_samples_operators)

    if kernel_type == "laplacian":
        return sample_bernoulli_kernel_operators(key, p_laplacian(sigma), m, n_samples_operators)

    if kernel_type == "polynomial_2":
        return sample_polynomial_kernel_operators(key, m, n_samples_operators, degree=degree, c=c)

    if kernel_type == "polynomial_1":
        return sample_polynomial_1_kernel_operators(key, m, n, n_samples_operators, degree=degree, c=c)

    if kernel_type == "low_order":
        return sample_low_order_kernel_operators(key, m, n_samples_operators, max_weight=max_weight)

    if kernel_type == "data_biased_low_order":
        if mode_probs is None:
            raise ValueError("mode_probs must be provided for data_biased_low_order.")
        return sample_data_biased_low_order_kernel_operators(
            key, mode_probs, m, n_samples_operators, max_weight=max_weight
        )

    raise ValueError(f"Unknown kernel_type: {kernel_type}")




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


@partial(jax.jit, static_argnames=("m", "n", "n_samples_operators", "kernel_type", "degree", "max_weight"))
def MMD_loss_classical_dataset(
    model_dataset: Array,
    target_dataset: Array,
    sigma: float,
    m: int,
    n: int,
    key: Array,
    n_samples_operators: int,
    mode_probs: Optional[Array],
    kernel_type: str = "gaussian",
    degree: int = 2,
    c: float = 1.0,
    max_weight: int = 2,
) -> Array:
    
    """ Estimate MMD^2 between two classical datasets using sampled Walsh observables """
    K = sample_kernel_operators(
        key=key,
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

    model_expectations = dataset_expectations(K, model_dataset.astype(jnp.float64))
    target_expectations = dataset_expectations(K, target_dataset.astype(jnp.float64))
    return jnp.real(jnp.mean((model_expectations - target_expectations) ** 2))


# ALTERNATIVE ESTIMATOR

@partial(jax.jit, static_argnames=("num_samples",))
def estimate_expectation(key: Array, T: Array, Z_diag: Array, num_samples: int) -> Array:
    """Estimate a Walsh expectation from sampled dataset entries."""
    M = T.shape[0]
    _, sk = jax.random.split(key)
    idx = jax.random.randint(sk, shape=(num_samples,), minval=0, maxval=M)
    xs = T[idx]
    dots = jnp.sum(xs * Z_diag, axis=1) % 2.0
    return jnp.mean(jnp.where(dots == 0, 1.0, -1.0))


@partial(jax.jit, static_argnames=("m", "num_photons", "num_samples1", "num_samples2", "num_samples3"))
def MMD_loss_alternative(
    params: Array,
    target_dataset: Array,
    sigma: float,
    m: int,
    num_photons: int,
    key: Array,
    num_samples1: int,
    num_samples2: int,
    num_samples3: int,
) -> Array:
    
    """Alternative MMD estimator using sampled bitstrings and root samples."""
    p = p_sigma(sigma)
    key_a, key_x, key_target = jax.random.split(key, 3)

    U = clements_unitary(params, m)
    A_samples = sample_bitstrings(key_a, m, p, num_samples1)

    s = jnp.ones(num_photons, dtype=int)
    X = sample_roots_dataset(key_x, s, num_samples2)

    def compute_single_term(a):
        z_diag = (1.0 - 2.0 * a).astype(U.dtype)
        weighted_U = z_diag[:, None] * U
        B = (jnp.conj(U.T) @ weighted_U)[:num_photons, :num_photons]

        Gs = generalized_glynn_on_dataset(B, s, X)
        circuit_exp = jnp.mean(Gs)
        circuit_exp_sq = unbiased_product_estimate(Gs)

        target_exp = estimate_expectation(key_target, target_dataset, 1.0 - 2.0 * a, num_samples3)
        return circuit_exp_sq - 2.0 * circuit_exp * target_exp

    terms = jax.vmap(compute_single_term)(A_samples)
    return jnp.real(jnp.mean(terms))
