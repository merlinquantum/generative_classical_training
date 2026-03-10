import jax.numpy as jnp
import jax
import numpy as np
import inspect
import os
from functools import partial
from src.helpers.circuit import butterfly_mzi_pattern, butterfly_unitary, clements_mzi_pattern, clements_unitary, mzi3_unitary, haar_unitary
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from src.models.training import Trainer

def close_to_identity_haar_init(m, init_key, max_value_perturb = 0.01):
    """
    Generate an m x m unitary that is close to identity from the Haar param
    :param m: number of modes
    """
    size = 2 * m * m
    params = np.zeros(size)

    # Set 1s for identity
    for i in range(m):
        params[i * (m + 1)] = 1

    # Add perturbation
    perturb = jax.random.uniform(init_key, shape=(size,), minval=0.0, maxval=max_value_perturb)
    for i in range(size):
        params[i] += perturb[i]

    return haar_unitary(params, m), params
    

def close_to_identity_clements_init(m, init_key, max_value_theta = 0.01, max_value_phi = 0.01, max_value_gamma = 0.01):
    """
    Generate an m x m unitary from a rectangular MZI mesh that is close to identity and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=clements_mzi_pattern(m)
    shape_params = jnp.shape(pairs)
    n_rows = shape_params[0]

    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    key3, init_key3 = jax.random.split(key2, 2)
    
    # theta values: small positive values near 0
    col1 = jax.random.uniform(init_key1, shape=(n_rows,), minval=0.0, maxval=max_value_theta)
    
    # phi values: small positive values near 0
    col2 = jax.random.uniform(init_key2, shape=(n_rows,), minval=0.0, maxval=max_value_phi)
    
    # Combine into final array
    params = jnp.column_stack((col1, col2))

    # Now generate gamma array
    gammas = jax.random.uniform(init_key3, shape=(m,), minval=0.0, maxval=max_value_gamma)
    
    return clements_unitary(params, gammas, m), params, gammas


def close_to_identity_mzi3_init(m, init_key, max_value_theta = 0.01, max_value_phi = 0.01, max_value_gamma = 0.01):
    """
    Generate an m x m unitary from a rectangular 3MZI mesh that is close to identity and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=clements_mzi_pattern(m)
    shape_params = jnp.shape(pairs)
    n_rows = shape_params[0]

    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    key3, init_key3 = jax.random.split(key2, 2)
    
    # theta values: small positive values near 0
    col1 = jax.random.uniform(init_key1, shape=(n_rows,), minval=jnp.pi/2 - max_value_theta/2, maxval=jnp.pi/2 + max_value_theta/2)
    
    # phi values: small positive values near 0
    col2 = jax.random.uniform(init_key2, shape=(n_rows,), minval=jnp.pi/2 - max_value_phi/2, maxval=jnp.pi/2 + max_value_phi/2)
    
    # Combine into final array
    params = jnp.column_stack((col1, col2))

    # Now generate gamma array
    gammas = jax.random.uniform(init_key3, shape=(m,), minval=0.0, maxval=max_value_gamma)
    
    return mzi3_unitary(params, gammas, m), params, gammas


def close_to_identity_butterfly_init(m, init_key, max_value_theta = 0.01, max_value_phi = 0.01, max_value_gamma = 0.01):
    """
    Generate an m x m unitary from a butterfly MZI mesh that is close to identity and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=clements_mzi_pattern(m)
    shape_params = jnp.shape(pairs)
    n_rows = shape_params[0]

    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    key3, init_key3 = jax.random.split(key2, 2)
    
    # theta values: small positive values near 0
    col1 = jax.random.uniform(init_key1, shape=(n_rows,), minval=0.0, maxval=max_value_theta)
    
    # phi values: small positive values near 0
    col2 = jax.random.uniform(init_key2, shape=(n_rows,), minval=0.0, maxval=max_value_phi)
    
    # Combine into final array
    params = jnp.column_stack((col1, col2))

    # Now generate gamma array
    gammas = jax.random.uniform(init_key3, shape=(m,), minval=0.0, maxval=max_value_gamma)
    
    return butterfly_unitary(params, gammas, m), params, gammas


def random_haar_init(m, init_key, max_value = 1.0):
    size = 2 * m * m
    params = jax.random.uniform(init_key, shape=(size,), minval=0.0, maxval=max_value)

    return haar_unitary(params, m), params


def random_clements_init(m, init_key):
    """
    Generate a random m x m unitary from a rectangular MZI mesh and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=clements_mzi_pattern(m)
    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    
    params=jax.random.uniform(init_key1, shape=jnp.shape(pairs), minval=0.0, maxval=2 * jnp.pi)
    gammas=jax.random.uniform(init_key2, shape=(m,), minval=0.0, maxval=2 * jnp.pi)
    
    return clements_unitary(params, gammas, m), params, gammas


def random_butterfly_init(m, init_key):
    """
    Generate a random m x m unitary from a butterfly MZI mesh and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=butterfly_mzi_pattern(m)
    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    
    params=jax.random.uniform(init_key1, shape=jnp.shape(pairs), minval=0.0, maxval=2 * jnp.pi)
    gammas=jax.random.uniform(init_key2, shape=(m,), minval=0.0, maxval=2 * jnp.pi)
    
    return butterfly_unitary(params, gammas, m), params, gammas


def random_mzi3_init(m, init_key):
    """
    Generate a random m x m unitary from a rectangular 3MZI mesh and returns the parameters that created it.

    :param m: number of modes
    """
    pairs=clements_mzi_pattern(m)
    key1, init_key1 = jax.random.split(init_key, 2)
    key2, init_key2 = jax.random.split(key1, 2)
    
    params=jax.random.uniform(init_key1, shape=jnp.shape(pairs), minval=0.0, maxval=2 * jnp.pi)
    gammas=jax.random.uniform(init_key2, shape=(m,), minval=0.0, maxval=2 * jnp.pi)
    
    return mzi3_unitary(params, gammas, m), params, gammas


def uniform_samples(init_key, num_samples):
    return jax.random.uniform(init_key,(num_samples,2), minval=0.0, maxval=2 * jnp.pi)

def VMF_samples(theta0, phi0, kappa, num_samples):
    """
    Draw samples from a von Mises–Fisher (vMF) distribution on the unit sphere S^2.
    theta0 and phi0 define the mean direction of the distribution in spherical coordinates (where the gaussian will be centered around)
    kappa is the concentration parameter of the distribution (approximately kappa = 1/sigma^2)
        kappa = 0 → uniform on the sphere
        larger kappa → samples more concentrated around theta0,phi0
    The implementation comes from paper by Carlos Pinzón and Kangsoo Jung "Fast Python sampler for the von Mises Fisher distribution"
    Source : https://hal.science/hal-04004568
    """
    mu = spherical_to_cartesian(theta0, phi0)
    samples = random_VMF(mu, kappa, size=(num_samples,))
    return cartesian_to_spherical(samples)


def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to a 3D unit vector.

    theta: polar angle [0, 2pi]
    phi: azimuthal angle [0, 2pi]
    """
    theta = jnp.mod(theta, 2 * jnp.pi)
    x = jnp.sin(theta / 2) * jnp.cos(phi)
    y = jnp.sin(theta / 2) * jnp.sin(phi)
    z = jnp.cos(theta / 2)
    return jnp.array([x, y, z])


def cartesian_to_spherical(v):
    """
    Convert a 3D Cartesian unit vector to spherical coordinates (theta, phi).
    """
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    r = jnp.sqrt(x ** 2 + y ** 2)
    theta = 2 * jnp.arctan2(r, z)

    phi = jnp.arctan2(y, x)
    phi = jnp.mod(phi, 2 * jnp.pi)  # ensure in [0, 2π)

    return jnp.stack([theta, phi], axis=-1)


def random_VMF(mu, kappa, size=None):
    """
    Von Mises - Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Source : https://hal.science/hal-04004568
    """
    # parse input parameters
    n = 1 if size is None else np.prod(size)
    shape = () if size is None else tuple(np.ravel(size))
    mu = np.asarray(mu)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape
    # z component : radial samples p e r p e n d i c u l a r to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles ( in cos and sin form )
    cos = random_VMF_angle(kappa, n)
    sin = np.sqrt(1 - cos ** 2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    return x.reshape((*shape, d))


def random_VMF_angle(k: float, n: int, d=3):
    """
    Generate n iid samples t with density function given by
    p ( t ) = some Constant * (1 - t ** 2 ) **(( d - 3 ) / 2 ) * exp ( kappa * t )
    Source : https://hal.science/hal-04004568
    """
    alpha = (d - 1) / 2
    t0 = r0 = np.sqrt(1 + (alpha / k) ** 2) - alpha / k
    log_t0 = k * t0 + (d - 1) * np.log(1 - r0 * t0)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        t = np.random.beta(alpha, alpha, m)
        t = 2 * t - 1
        t = (r0 + t) / (1 + r0 * t)
        log_acc = k * t + (d - 1) * np.log(1 - r0 * t) - log_t0
        t = t[np.random.random(m) < np.exp(log_acc)]
        out.append(t)
        found += len(out[- 1])
    return np.concatenate(out)[:n]


def butterfly_init(m, distribution, *distribution_params):
    """
    Initialize a butterfly unitary with parameters sampled from a given distribution.
    The distribution function must accept a keyword argument `num_samples`, which
    will be set automatically to match the number of MZIs.

    Parameters
    ----------
    m : int
        Number of modes in the butterfly mesh.
    distribution : callable
        A sampling function that generates parameters. It must have a
        `num_samples` parameter in its signature.
    *distribution_params : tuple
        Positional arguments to pass to the `distribution` function
        (excluding `num_samples`).

    Returns
    -------
    U : jnp.ndarray
        The butterfly unitary of shape `(m, m)` generated with sampled parameters.
    params : jnp.ndarray
        The sampled parameters from the given distribution. The shape depends on
        the output of `distribution`.

    Sample Usage
    ------
    initialize butterfly circuit with uniformly sampled parameters:
    butterfly_init(8, uniform_samples, jax.random.PRNGKey(0))

    initialize butterfly circuit with identity intialization (theta's centered around 0 and phi's uniform) with kappa=100:
    butterfly_init(8, VMF_samples, 0, 0, 100)

    """

    # Check that distribution has a parameter called num_samples
    sig = inspect.signature(distribution)
    if "num_samples" not in sig.parameters:
        raise TypeError(f"Distribution {distribution.__name__} must have a 'num_samples' parameter.")

    pairs=butterfly_mzi_pattern(m)
    num_mzi=jnp.shape(pairs)[0]
    params=distribution(*distribution_params, num_samples=num_mzi)
    gammas=jnp.zeros(m)
    return butterfly_unitary(params, gammas, m), params, gammas

def clements_init(m, distribution, *distribution_params):

    # Check that distribution has a parameter called num_samples
    sig = inspect.signature(distribution)
    if "num_samples" not in sig.parameters:
        raise TypeError(f"Distribution {distribution.__name__} must have a 'num_samples' parameter.")

    pairs=clements_mzi_pattern(m)
    num_mzi = jnp.shape(pairs)[0]
    params = distribution(*distribution_params, num_samples=num_mzi)
    gammas=jnp.zeros(m)
    return clements_unitary(params, gammas, m), params, gammas

def mzi3_init(m, distribution, *distribution_params):

    # Check that distribution has a parameter called num_samples
    sig = inspect.signature(distribution)
    if "num_samples" not in sig.parameters:
        raise TypeError(f"Distribution {distribution.__name__} must have a 'num_samples' parameter.")

    pairs=clements_mzi_pattern(m)
    num_mzi = jnp.shape(pairs)[0]
    params = distribution(*distribution_params, num_samples=num_mzi)
    gammas=jnp.zeros(m)
    return mzi3_unitary(params, gammas, m), params, gammas


#@partial(jit, static_argnames={"num_photons"})
def single_mode_marginals(U: jnp.array, num_photons: int):
    """
    Returns vector whose jth entry contains the probability of measuring exactly 1 photon in the jth mode
    from unitary U with num_photons input photons |1,1,1,...num_photons,0,0,0>
    """
    U = U[:, :num_photons]
    modulus_sq = jnp.real(U * jnp.conj(U))  # element-wise |U|^2
    print(modulus_sq)
    col_sums = jnp.sum(modulus_sq, axis=1)  # sum over cols
    print(col_sums)
    return col_sums / num_photons


def compute_target_marginals(data):
    """
    Returns single mode marginals of a target dataset. (takes mean over the dataset for each feature individually)
    """
    feature_means = np.mean(data, axis=0)
    return feature_means


def L2_loss_clements(params, target_marginals,m,num_photons):
    """
    L2 loss to compare single_mode_marginals of a unitary to a target marginal distribution.
    """
    U=clements_unitary(params,m)
    U_marginals=single_mode_marginals(U,num_photons)
    
    return jnp.sum((target_marginals-U_marginals)**2)