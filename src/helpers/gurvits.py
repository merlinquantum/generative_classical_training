import jax.numpy as jnp
import jax.scipy.special as jsp
import jax
jax.config.update("jax_enable_x64", True)


def sample_Z_dataset(key, n, num_samples):
    """
    Returns Z of shape (num_samples, k),
    where each row is a {-1, 1}^m bitstring
    """
    binary_samples = jax.random.randint(key, shape=(num_samples, n), minval=0, maxval=2)
    # Map 0 -> -1 and 1 -> 1
    samples = 2 * binary_samples - 1
    return samples


@jax.jit
def glynn_given_z(A, z):
    """
    One Glynn estimator sample for a fixed z bitstring and matrix A
    """
    return jnp.prod(z) * jnp.prod(A @ z)


@jax.jit
def glynn_on_dataset(A, Z):
    """
    Computes the Glynn estimator for each sample in the Z bistring dataset and returns a dataset of these Glynn samples
    """
    return jax.vmap(lambda z: glynn_given_z(A, z))(Z)


def unbiased_product_estimate(Gs):
    M = Gs.shape[0]
    numerator = jnp.sum(Gs)**2 - jnp.sum(Gs**2)
    return numerator / (M * (M - 1))

