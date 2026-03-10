import jax.numpy as jnp
import jax
from jax import jit, lax
from jaxopt import LBFGS
from functools import partial
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=("m",))
def haar_unitary(params: jnp.ndarray, m: int) -> jnp.ndarray:
    # params is assumed length 2*m*m: first real part, then imag part
    u = (params[: m*m].reshape(m, m)
         + 1j * params[m*m : 2*m*m].reshape(m, m))

    q, r = jnp.linalg.qr(u)

    # Phase correction from R's diagonal: d = diag(r) / |diag(r)|
    d = jnp.diag(r)
    d = jnp.where(jnp.abs(d) > 0, d / jnp.abs(d), 1.0 + 0j)

    # Multiply Q by diag(conj(d)) on the right
    return q @ jnp.diag(jnp.conj(d))


@partial(jit, static_argnames={"m"})
def clements_unitary(params: jnp.ndarray, gammas: jnp.ndarray, m: int, pairs=None) -> jnp.ndarray:
    """
    Compute the m x m unitary of a Clements interferometer given phase values.

    :param params: values of the parameters for the phases of the interferometer.
    :param gammas: final layer of phases for completion of the unitary
    :param m: number of modes
    """
    def scan_step(U, data):
        (theta, phi), (i, j) = data
        B = bs(theta, phi)
        U = apply_bs_to_rows(U, i, j, B)
        return U, None

    # prepare parameters and pair list
    if pairs is None:
        pairs = clements_mzi_pattern(m)
    
    data = (params, pairs)

    # initialize identity matrix
    U0 = jnp.eye(m, dtype=jnp.complex64)
    
    #lax.scan uses functional recursive programming to build the unitary from MZI transformations
    U_mesh, _ = lax.scan(scan_step, U0, data, length=params.shape[0])

    #Apply layer of phases gammas
    U_final = (jnp.exp(1j * gammas)[:, None]) * U_mesh
    
    return U_final

    
def clements_mzi_pattern(m: int) -> jnp.ndarray:
    """
    Generate the list of (i, j) mode pairs for a Clements rectangular mesh over m modes.

    :param m: number of modes
    """
    pattern = []
    for layer in range(m):
        start = layer % 2  # even: (0,1),(2,3)... ; odd: (1,2),(3,4)...
        for i in range(start, m - 1, 2):
            pattern.append((i, i + 1))
    return jnp.array(pattern)


def bs(theta: float, phi: float) -> jnp.ndarray:
    """
    Create a 2x2 MZI interferometer unitary from theta and phi.
    """
    return jnp.array([
        [jnp.exp(1j * phi) * jnp.cos(theta), -jnp.sin(theta)],
        [jnp.exp(1j * phi) * jnp.sin(theta), jnp.cos(theta)]
    ], dtype=jnp.complex64)


def apply_bs_to_rows(U: jnp.ndarray, i: int, j: int, B: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a 2x2 beam splitter matrix B to rows i and j of U, as in the Clements decomposition.
    """
    assert jnp.iscomplexobj(U), "U is not dtype=jnp.complex64!"
    
    Ui = U[i, :]
    Uj = U[j, :]
    new_i = B[0, 0] * Ui + B[0, 1] * Uj
    new_j = B[1, 0] * Ui + B[1, 1] * Uj
    U = U.at[i, :].set(new_i)
    U = U.at[j, :].set(new_j)
    return U


# BUTTERFLY MESH
# butterfly_stride_pattern, pairs_for_stride, and all_pairs_from_strides are all helper functions to generate the butterfly mesh
# butterfly_mzi_pattern(m) is analogous to clements_mzi_pattern(m)
# butterfly_unitary(params,m) is analogous to clements_unitary(params, m) 
# random_butterfly_init is analgous to random_clements_init

def butterfly_stride_pattern(m: int) -> jnp.ndarray:
    """Return the stride pattern as a JAX array of length m-1 (JIT-safe)."""

    num_layers = m - 1
    strides = jnp.zeros((num_layers,), dtype=jnp.int32)

    def body_fun(k, arr):
        # Binary expansion trick to generate pattern recursively
        # For each layer index k, compute stride as the largest power of 2 dividing (k+1)
        stride = (k + 1) & -(k + 1)  # isolates lowest set bit
        return arr.at[k].set(stride)

    strides = jax.lax.fori_loop(0, num_layers, body_fun, strides)
    return strides


def pairs_for_stride(s, m):
    """
    Generate exactly m/2 disjoint pairs (i, i+s) for a given stride s.
    """
    num_pairs = m // 2
    k = jnp.arange(num_pairs)       # pair index 0..m/2-1
    j = k % s                       # offset cycles 0..s-1
    block = k // s                  # which block
    i_vals = j + 2 * s * block      # left index
    j_vals = i_vals + s             # right index
    return jnp.stack([i_vals, j_vals], axis=-1)  # shape (m/2, 2)


def all_pairs_from_strides(strides, m):
    """
    Given a vector of strides, return all (i,j) pairs for each stride,
    concatenated into one array. Each layer contributes exactly m/2 pairs.
    """
    layers = jax.vmap(lambda s: pairs_for_stride(s, m))(strides)  # (num_layers, m/2, 2)
    return layers.reshape(-1, 2)  # (num_layers*m/2, 2)


def butterfly_mzi_pattern(m):
    return all_pairs_from_strides(butterfly_stride_pattern(m),m)


@partial(jit, static_argnames={"m"})
def butterfly_unitary(params: jnp.ndarray, gammas: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    Compute the m x m unitary of a Butterfly interferometer given phase values.

    :param params: values of the parameters for the phases of the interferometer.
    :param gammas: final layer of phases for completion of the unitary
    :param m: number of modes
    """
    params=params.reshape((-1, 2))
    
    def scan_step(U, data):
        (theta, phi), (i, j) = data
        B = bs(theta, phi)
        U = apply_bs_to_rows(U, i, j, B)
        return U, None

    # initialize identity matrix
    U0 = jnp.eye(m, dtype=jnp.complex64)

    # prepare parameters and pair list
    data = (params, butterfly_mzi_pattern(m))
    
    #lax.scan uses functional recursive programming to build the unitary from MZI transformations
    U_mesh, _ = lax.scan(scan_step, U0, data, length=params.shape[0])

    U_final = (jnp.exp(1j * gammas)[:, None]) * U_mesh
    
    return U_final


### 3MZI MESH

def bs3(theta: float, phi: float) -> jnp.ndarray:
    """
    Create a 2x2 3MZI unitary from theta and phi.
    """
    bs3_matrix = jnp.array([
        [jnp.exp(1j * phi) * jnp.sin(theta/2) + 1j * jnp.cos(theta/2),
        1j * jnp.exp(1j * phi) * jnp.sin(theta/2) + jnp.cos(theta/2)],
        [jnp.exp(1j * phi) * jnp.cos(theta/2) - 1j * jnp.sin(theta/2),
         1j * jnp.exp(1j * phi) * jnp.cos(theta/2) - jnp.sin(theta/2) ]
    ], dtype=jnp.complex64)/jnp.sqrt(2)
    
    return bs3_matrix


@partial(jit, static_argnames={"m"})
def mzi3_unitary(params: jnp.ndarray, gammas: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    Compute the m x m unitary of a 3MZI interferometer given phase values.

    :param params: values of the parameters for the phases of the interferometer.
    :param gammas: final layer of phases for completion of the unitary
    :param m: number of modes
    """
    params = params.reshape((-1, 2))

    def scan_step(U, data):
        (theta, phi), (i, j) = data
        B = bs3(theta, phi)
        U = apply_bs_to_rows(U, i, j, B)
        return U, None

    # initialize identity matrix
    U0 = jnp.eye(m, dtype=jnp.complex64)

    # prepare parameters and pair list
    data = (params, clements_mzi_pattern(m))

    # lax.scan uses functional recursive programming to build the unitary from MZI transformations
    U_mesh, _ = lax.scan(scan_step, U0, data, length=params.shape[0])

    U_final = (jnp.exp(1j * gammas)[:, None]) * U_mesh

    return U_final



