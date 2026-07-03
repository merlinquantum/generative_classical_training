import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray




# ************************************ Kernel Probabilities ********************************************************

def p_sigma(sigma: float) -> Array:
    """Bernoulli parameter for the Gaussian kernel."""
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


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
