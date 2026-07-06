import functools
import inspect
from typing import Callable, Iterable, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from src.helpers.circuit import (
    butterfly_mzi_pattern,
    butterfly_unitary,
    clements_mzi_pattern,
    clements_unitary,
    haar_unitary,
    mzi3_unitary,
)

# Precision of JAX fixed at 64 bits
jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray

#Global variable
TWO_PI = 2.0 * jnp.pi


# Generic helpers

def pattern_and_unitary(ansatz: str) -> Tuple[Callable[[int], Array], Callable]:
    """Return the MZI pattern and unitary function for a mesh ansatz."""
    if ansatz == "clements":
        return clements_mzi_pattern, clements_unitary
    if ansatz == "butterfly":
        return butterfly_mzi_pattern, butterfly_unitary
    if ansatz == "mzi3":
        return clements_mzi_pattern, mzi3_unitary
    raise ValueError(f"Unknown mesh ansatz: {ansatz}")


def split3(key: Array) -> Tuple[Array, Array, Array]:
    """Split a JAX random key into three independent keys"""
    return tuple(jax.random.split(key, 3))


def mesh_random_params(m: int, key: Array, ansatz: str) -> Tuple[Array, Array]:
    """Generate uniformly random mesh parameters and output phases"""
    pattern_fn, _ = pattern_and_unitary(ansatz)
    pairs = pattern_fn(m)
    key_params, key_gammas = jax.random.split(key, 2)
    params = jax.random.uniform(key_params, shape=pairs.shape, minval=0.0, maxval=TWO_PI) # theta, phi
    gammas = jax.random.uniform(key_gammas, shape=(m,), minval=0.0, maxval=TWO_PI)
    return params, gammas


def mesh_normal_params(m: int, key: Array, ansatz: str) -> Tuple[Array, Array]:
    """Gerenate normally distributed mesh parameters and output phases"""
    pattern_fn, _ = pattern_and_unitary(ansatz)
    pairs = pattern_fn(m)
    key_params, key_gammas = jax.random.split(key, 2)
    params = jax.random.normal(key_params, shape=pairs.shape)
    gammas = jax.random.normal(key_gammas, shape=(m,))
    return params, gammas


def mesh_close_to_identity_params(m: int,key: Array,ansatz: str,max_value_theta: float = 0.01,max_value_phi: float = 0.01,max_value_gamma: float = 0.01) -> Tuple[Array, Array]:
    """Generate mesh parameters corresponding to a unitary close to identity"""
    pattern_fn, _ = pattern_and_unitary(ansatz)
    pairs = pattern_fn(m)
    n_rows = pairs.shape[0]
    key_theta, key_phi, key_gamma = split3(key)

    if ansatz == "mzi3":
        # theta ~ pi/2 ± small perturbation
        theta = jax.random.uniform(
            key_theta,
            shape=(n_rows,),
            minval=jnp.pi / 2 - max_value_theta / 2,
            maxval=jnp.pi / 2 + max_value_theta / 2,
        )
        # phi ~ pi/2 ± small perturbation
        phi = jax.random.uniform(
            key_phi,
            shape=(n_rows,),
            minval=jnp.pi / 2 - max_value_phi / 2,
            maxval=jnp.pi / 2 + max_value_phi / 2,
        )

    else:
        theta = jax.random.uniform(key_theta, shape=(n_rows,), minval=0.0, maxval=max_value_theta)
        phi = jax.random.uniform(key_phi, shape=(n_rows,), minval=0.0, maxval=max_value_phi)

    params = jnp.stack([theta, phi], axis=1)
    gammas = jax.random.uniform(key_gamma, shape=(m,), minval=0.0, maxval=max_value_gamma)
    return params, gammas


def haar_unitary_with_gammas(params: Array, gammas: Array, m: int) -> Array:
    """Wrapper used when a generic function expects params and gammas"""
    return haar_unitary(params, m)


# INITIALIZATION STRATEGIES:

# ************************ Close-to-identity initialization ***************************************************************
def close_to_identity_haar_init(m: int, init_key: Array, max_value_perturb: float = 0.01) -> Tuple[Array, Array]:
    """
    Generate an m x m unitary that is close to identity from the Haar param
    :param m: number of modes
    """    
    size = 2 * m * m

    #We set the real diagonal of the flattened matrix
    params = jnp.zeros(size, dtype=jnp.float64)
    diag_idx = jnp.arange(m) * (m + 1)
    params = params.at[diag_idx].set(1.0)

    # Add perturbation
    perturb = jax.random.uniform(init_key, shape=(size,), minval=0.0, maxval=max_value_perturb)
    params = params + perturb
    return haar_unitary(params, m), params


def close_to_identity_mesh_init(m: int,init_key: Array,ansatz: str,max_value_theta: float = 0.01,max_value_phi: float = 0.01,max_value_gamma: float = 0.01) -> Tuple[Array, Array, Array]:
    pattern_fn, unitary_fn = pattern_and_unitary(ansatz)
    params, gammas = mesh_close_to_identity_params(
        m=m,
        key=init_key,
        ansatz=ansatz,
        max_value_theta=max_value_theta,
        max_value_phi=max_value_phi,
        max_value_gamma=max_value_gamma,
    )
    return unitary_fn(params, gammas, m), params, gammas

# We initialize a {Clements, butterfly, MZI3} mesh close to the identity :

def close_to_identity_clements_init(m: int, init_key: Array, **kwargs):
    return close_to_identity_mesh_init(m, init_key, ansatz="clements", **kwargs)


def close_to_identity_butterfly_init(m: int, init_key: Array, **kwargs):
    return close_to_identity_mesh_init(m, init_key, ansatz="butterfly", **kwargs)


def close_to_identity_mzi3_init(m: int, init_key: Array, **kwargs):
    return close_to_identity_mesh_init(m, init_key, ansatz="mzi3", **kwargs)


def close_to_identity_init_general(m: int, init_key: Array, ansatz: str = "haar"):
    """Generic close-to-identity initialization for Haar or mesh ansatz"""
    if ansatz == "haar":
        U, params = close_to_identity_haar_init(m, init_key)
        return U, params, jnp.zeros(m)
    if ansatz in {"clements", "butterfly", "mzi3"}:
        return close_to_identity_mesh_init(m, init_key, ansatz=ansatz)
    raise ValueError(f"Unknown ansatz: {ansatz}")


# *********************************** Random Initialization *****************************************************

def random_init(m: int, init_key: Array, max_value: float = 1.0) -> Tuple[Array, Array]:
    """Initialize a Haar unitary with uniformly random parameters"""
    size = 2 * m * m
    params = jax.random.uniform(init_key, shape=(size,), minval=0.0, maxval=max_value)
    return haar_unitary(params, m), params


def random_mesh_init(m: int, init_key: Array, ansatz: str):
    """Initialize a mesh unitary with uniformly random parameters"""
    _, unitary_fn = pattern_and_unitary(ansatz)
    params, gammas = mesh_random_params(m, init_key, ansatz)
    return unitary_fn(params, gammas, m), params, gammas


# Random initialization for all the ansatz:

def random_clements_init(m: int, init_key: Array):
    return random_mesh_init(m, init_key, ansatz="clements")


def random_butterfly_init(m: int, init_key: Array):
    return random_mesh_init(m, init_key, ansatz="butterfly")


def random_mzi3_init(m: int, init_key: Array):
    return random_mesh_init(m, init_key, ansatz="mzi3")


def random_init_general(m: int, init_key: Array, ansatz: str = "haar"):
    """Generic random initialization for Haar or mesh ansatz."""
    if ansatz == "haar":
        U, params = random_init(m, init_key)
        return U, params, jnp.zeros(m)
    if ansatz in {"clements", "butterfly", "mzi3"}:
        return random_mesh_init(m, init_key, ansatz=ansatz)
    raise ValueError(f"Unknown ansatz: {ansatz}")


#*************************************** Weak Unbiased Initialization  ****************************************

def unbiased_init(m: int, init_key: Array) -> Tuple[Array, Array]:
    """Initialize a Haar unitary with normally distributed parameters"""
    size = 2 * m * m
    params = jax.random.normal(init_key, shape=(size,))
    return haar_unitary(params, m), params


def unbiased_mesh_init(m: int, init_key: Array, ansatz: str):
    """Initialize a mesh unitary with normally distributed parameters"""
    _, unitary_fn = pattern_and_unitary(ansatz)
    params, gammas = mesh_normal_params(m, init_key, ansatz)
    return unitary_fn(params, gammas, m), params, gammas

# Unbiased normal initialization for all the ansatz:

def unbiased_clements_init(m: int, init_key: Array):
    return unbiased_mesh_init(m, init_key, ansatz="clements")


def unbiased_butterfly_init(m: int, init_key: Array):
    return unbiased_mesh_init(m, init_key, ansatz="butterfly")


def unbiased_mzi3_init(m: int, init_key: Array):
    return unbiased_mesh_init(m, init_key, ansatz="mzi3")


def unbiased_init_general(m: int, init_key: Array, ansatz: str = "haar"):
    """Generic unbiased initialization for Haar or mesh ansatz"""
    if ansatz == "haar":
        U, params = unbiased_init(m, init_key)
        return U, params, jnp.zeros(m)
    if ansatz in {"clements", "butterfly", "mzi3"}:
        return unbiased_mesh_init(m, init_key, ansatz=ansatz)
    raise ValueError(f"Unknown ansatz: {ansatz}")


#************************************ Pairwise Initialization ************************************************

def pairwise_uniform_unitary(m: int, n_pairs: int) -> Array:
    """
    Build a block-diagonal unitary that applies a 50:50 beam splitter
    on pairs (0,1), (2,3), ..., (2*n_pairs-2, 2*n_pairs-1),
    and identity on the remaining modes.

    This is adapted to pairwise-choice datasets with one photon per pair !!
    """    
    U = jnp.eye(m, dtype=jnp.complex128)
    H = (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128)

    for p in range(n_pairs):
        i = 2 * p
        U = U.at[i : i + 2, i : i + 2].set(H)
    return U


def pairwise_uniform_haar_init(m: int, init_key: Array, n_pairs: int):
    """Initialize the Haar ansatz with a pairwise-uniform target unitary"""
    U_target = pairwise_uniform_unitary(m, n_pairs)
    params = jnp.concatenate([jnp.real(U_target).reshape(-1), jnp.imag(U_target).reshape(-1)])
    return haar_unitary(params, m), params, jnp.zeros(m)


def find_pairwise_mzi_rows(pairs: Array, n_pairs: int) -> np.ndarray:
    """
    Find which MZI rows correspond to the pairwise blocks.

    For a pairwise dataset, the active blocks are:
        (0, 1), (2, 3), (4, 5), ...

    In a mesh ansatz such as Clements or Butterfly, the parameters are not
    stored directly by block number. Instead, the array `pairs` tells us which
    two modes are mixed by each MZI row.

    This function returns the first MZI row found for each pairwise block.
    We use only the first occurrence so that each block is initialized once.
    """
    target_pairs = [(2 * p, 2 * p + 1) for p in range(n_pairs)]

    selected_rows = []
    already_selected = set()

    for row_idx, pair in enumerate(np.asarray(pairs)):
        mode_1 = int(pair[0])
        mode_2 = int(pair[1])
        current_pair = tuple(sorted((mode_1, mode_2)))

        if current_pair in target_pairs and current_pair not in already_selected:
            selected_rows.append(row_idx)
            already_selected.add(current_pair)

    if len(selected_rows) != n_pairs:
        missing_pairs = set(target_pairs) - already_selected
        print(f"Warning: some pairwise blocks were not found in the mesh: {missing_pairs}")

    return np.asarray(selected_rows, dtype=np.int32)


def pairwise_uniform_mesh_init(m: int,init_key: Array,n_pairs: int,ansatz: str,theta_value: Optional[float] = None,phi_value: float = jnp.pi / 2,max_gamma: float = 0.0):
    """Initialize a mesh so that each pairwise block starts in a uniform superposition"""
    
    pattern_fn, unitary_fn = pattern_and_unitary(ansatz)
    pairs = pattern_fn(m)
    params = jnp.zeros(pairs.shape, dtype=jnp.float64)

    if ansatz == "mzi3":
        params = params.at[:, 0].set(jnp.pi / 2)
        params = params.at[:, 1].set(jnp.pi / 2)
        theta = jnp.pi / 2 if theta_value is None else theta_value
    else:
        theta = 0.0 if theta_value is None else theta_value

    active_rows = find_pairwise_mzi_rows(pairs, n_pairs)
    if active_rows.size > 0:
        params = params.at[active_rows, 0].set(theta)
        params = params.at[active_rows, 1].set(phi_value)

    if max_gamma > 0.0:
        gammas = jax.random.uniform(init_key, shape=(m,), minval=0.0, maxval=max_gamma)
    else:
        gammas = jnp.zeros(m)

    return unitary_fn(params, gammas, m), params, gammas

# Pairwise-uniform initialization for all the ansatz :

def pairwise_uniform_clements_init(m: int, init_key: Array, n_pairs: int, **kwargs):
    return pairwise_uniform_mesh_init(m, init_key, n_pairs, ansatz="clements", **kwargs)


def pairwise_uniform_butterfly_init(m: int, init_key: Array, n_pairs: int, **kwargs):
    return pairwise_uniform_mesh_init(m, init_key, n_pairs, ansatz="butterfly", **kwargs)


def pairwise_uniform_mzi3_init(m: int, init_key: Array, n_pairs: int, **kwargs):
    return pairwise_uniform_mesh_init(m, init_key, n_pairs, ansatz="mzi3", **kwargs)


def pairwise_uniform_init_general(m: int,init_key: Array,ansatz: str = "clements",n_pairs: int = 10):
    if ansatz == "haar":
        return pairwise_uniform_haar_init(m=m, init_key=init_key, n_pairs=n_pairs)
    if ansatz in {"clements", "butterfly", "mzi3"}:
        return pairwise_uniform_mesh_init(m=m, init_key=init_key, n_pairs=n_pairs, ansatz=ansatz)
    raise ValueError(f"Unknown ansatz: {ansatz}")


#**************************************** Block-uniform Initialization ***************************************

def fourier_matrix(d: int) -> Array:
    """Discrete Fourier transform matrix F_d"""
    a = jnp.arange(d)
    omega = jnp.exp(2j * jnp.pi / d)
    F = omega ** (a[:, None] * a[None, :])
    return (F / jnp.sqrt(d)).astype(jnp.complex128)


def block_uniform_unitary(m: int, block_sizes: Sequence[int]) -> Array:
    """Build a block-diagonal unitary:  U = F_{d_1} ⊕ F_{d_2} ⊕ ... ⊕ F_{d_l} ⊕ I_extra
    here the block_sizes = [d_1, d_2, ..., d_l]"""

    U = jnp.eye(m, dtype=jnp.complex128)
    start = 0
    for d in map(int, block_sizes):
        end = start + d
        U = U.at[start:end, start:end].set(fourier_matrix(d))
        start = end
    return U

# Input state |1,0,0, 1,0,0,0, 1,0, ...>
def block_first_modes(block_sizes: Sequence[int]) -> Array:
    """Input modes with one photon in the first mode of each categorical block"""
    starts = np.cumsum([0] + [int(d) for d in block_sizes[:-1]])
    return jnp.asarray(starts, dtype=jnp.int32)


def block_uniform_haar_init(m: int, init_key: Array, block_sizes: Sequence[int]):
    """Initialize the Haar ansatz with a block-uniform target unitary"""
    U_target = block_uniform_unitary(m, block_sizes)
    params = jnp.concatenate([jnp.real(U_target).reshape(-1), jnp.imag(U_target).reshape(-1)])
    return haar_unitary(params, m), params, jnp.zeros(m)


def block_uniform_init_general(m: int,init_key: Array,ansatz: str = "haar",block_sizes: Optional[Sequence[int]] = None):
    """Generic block-uniform initialization, currently implemented for Haar only"""
    if block_sizes is None:
        raise ValueError("block_sizes must be provided.")
    if ansatz == "haar":
        return block_uniform_haar_init(m=m, init_key=init_key, block_sizes=block_sizes)
    raise NotImplementedError("Block-uniform is currently implemented only for the Haar ansatz.")


# ************************************** Data-dependent Initialization **************************************************

def target_marginals(data: np.ndarray) -> np.ndarray:
    """Single-mode empirical marginals from the target dataset"""
    return np.mean(data, axis=0)

#Model marginals from modulus_sq matrix
#modulus_sq matrix is an (m x n) array whose elements are |U_{i,j}|^2 for i=0,1,...,m-1 and j=0,1,...,n-1
#for each row of the modulus_sq matrix compute all of the elementary symmetric polynomials and add them up in specific way to get that modes marginal

@functools.partial(jax.jit, static_argnames=("num_photons",))
def elementary_symmetric_polynomials_single_mode(modulus_sq_row: Array, num_photons: int) -> Array:
    """Compute elementary symmetric polynomials e_0, ..., e_n for one row."""
    polynomials = jnp.zeros((num_photons + 1,), dtype=modulus_sq_row.dtype).at[0].set(1.0)

    def scan_step(current_polynomials, x):
        def update_one(loop_index, updated_polynomials):
            k = num_photons - loop_index + 1
            return updated_polynomials.at[k].set(updated_polynomials[k] + x * updated_polynomials[k - 1])

        new_polynomials = lax.fori_loop(1, num_photons + 1, update_one, current_polynomials)
        return new_polynomials, None

    polynomials, _ = lax.scan(scan_step, polynomials, modulus_sq_row)
    return polynomials


@functools.partial(jax.jit, static_argnames=("num_photons",))
def model_marginals(modulus_sq: Array, num_photons: Optional[int] = None) -> Array:
    """Compute model one-mode marginals from |U[:, occupied_inputs]|^2"""
    if num_photons is None:
        num_photons = modulus_sq.shape[1]

    orders = jnp.arange(1, num_photons + 1)
    factorials = jnp.exp(lax.lgamma(orders + 1.0))
    signs = (-1.0) ** (orders + 1)

    def one_row(row):
        e = elementary_symmetric_polynomials_single_mode(row, num_photons)[1:]
        return jnp.sum(signs * factorials * e)

    return jax.vmap(one_row)(modulus_sq)


def modulus_sq_from_U_general(params: Array,gammas: Array,m: int,init_state_ind: Array,unitary_fn: Callable) -> Array:
    """Compute the squared modulus of the columns occupied by the input photons"""
    U = unitary_fn(params, gammas, m)
    return jnp.abs(U[:, init_state_ind]) ** 2


def model_marginals_from_params_general(params: Array,gammas: Array,m: int,init_state_ind: Array,num_photons: int,unitary_fn: Callable) -> Array:
    """Compute model marginals directly from trainable parameters"""
    modulus_sq = modulus_sq_from_U_general(params,gammas,m,init_state_ind,unitary_fn)
    return model_marginals(modulus_sq, num_photons=num_photons)


@functools.partial(jax.jit, static_argnames=("m", "num_photons", "unitary_fn"))
def loss_function_general(params: Array,gammas: Array,target_margs: Array,m: int,init_state_ind: Array,num_photons: int,unitary_fn: Callable) -> Array:
    """Compute the marginal-matching loss between model and target marginals"""
    model_margs = model_marginals_from_params_general(params,gammas,m,init_state_ind,num_photons,unitary_fn)
    return 0.5 * jnp.sum((model_margs - target_margs) ** 2)


@functools.partial(jax.jit, static_argnames=("m", "num_photons", "unitary_fn"))
def gradient_descent_step_general(params: Array,gammas: Array,target_margs: Array,learning_rate: float,m: int,init_state_ind: Array, num_photons: int,unitary_fn: Callable):
    """Perform one gradient descent step on params and gammas"""
    loss_value, (grad_params, grad_gammas) = jax.value_and_grad(
        loss_function_general,
        argnums=(0, 1),
    )(params, gammas, target_margs, m, init_state_ind, num_photons, unitary_fn)

    params = params - learning_rate * grad_params
    gammas = gammas - learning_rate * grad_gammas
    return params, gammas, loss_value


@functools.partial(jax.jit, static_argnames=("num_steps", "m", "num_photons", "unitary_fn"))
def train_general_scan(initial_params: Array,initial_gammas: Array,target_margs: Array,learning_rate: float,num_steps: int,m: int, init_state_ind: Array, num_photons: int,unitary_fn: Callable):
    """ Returns final params, gammas, and the loss history"""

    def step(carry, _):
        params, gammas = carry
        params, gammas, loss_value = gradient_descent_step_general(params,gammas,target_margs,learning_rate,m,init_state_ind,num_photons,unitary_fn)
        return (params, gammas), loss_value

    (params, gammas), losses = lax.scan(step, (initial_params, initial_gammas), xs=None, length=num_steps)
    return params, gammas, losses


def train_general(initial_params: Array,initial_gammas: Array, target_margs: Array,learning_rate: float,num_steps: int,m: int, init_state_ind: Array, num_photons: int,unitary_fn: Callable, print_every: Optional[int] = 250):
    """Train parameters to match target marginals and optionally print losses"""
    params, gammas, losses = train_general_scan(
        initial_params,
        initial_gammas,
        target_margs,
        learning_rate,
        num_steps,
        m,
        init_state_ind,
        num_photons,
        unitary_fn,
    )

    if print_every is not None:
        losses_host = np.asarray(losses)
        for step in range(0, num_steps, print_every):
            print(f"step={step}, loss={losses_host[step]}")
        print(f"step={num_steps - 1}, loss={losses_host[-1]}")

    return params, gammas, losses


def data_dependent_init_general(m: int,n: int,key: Array,X_train: np.ndarray,ansatz: str = "haar",init_state_ind: Optional[Array]=None, learning_rate: float = 0.5,num_steps: int = 10000,print_every: Optional[int] = 250):
    """Initialize parameters by pre-training them to match data marginals"""
    target_margs = jnp.asarray(target_marginals(X_train), dtype=jnp.float64)
    
    # We adapt the input state to block pairwwise dataset:
    if init_state_ind is None:
        init_state_ind = jnp.arange(n, dtype=jnp.int32)
    else:
        init_state_ind = jnp.asarray(init_state_ind, dtype=jnp.int32)

    num_photons = int(init_state_ind.shape[0])

    if ansatz == "haar":
        _, params, gammas = random_init_general(m, key, ansatz="haar")
        unitary_fn = haar_unitary_with_gammas
    elif ansatz in {"clements", "butterfly", "mzi3"}:
        _, params, gammas = random_init_general(m, key, ansatz=ansatz)
        _, unitary_fn = pattern_and_unitary(ansatz)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")

    params, gammas, losses = train_general(
        params,
        gammas,
        target_margs,
        learning_rate,
        num_steps,
        m,
        init_state_ind, 
        num_photons,
        unitary_fn,
        print_every=print_every,
    )

    U = unitary_fn(params, gammas, m)
    return U, params, gammas, losses




# DISTRIBUTION BASED MESH INITIALIZATION

def uniform_samples(init_key: Array, num_samples: int) -> Array:
    """Generate uniformly distributed angle pairs"""
    return jax.random.uniform(init_key, (num_samples, 2), minval=0.0, maxval=TWO_PI)


def spherical_to_cartesian(theta: Array, phi: Array) -> Array:
    """Convert spherical angles to a Cartesian unit vector"""
    theta = jnp.mod(theta, TWO_PI)
    x = jnp.sin(theta / 2) * jnp.cos(phi)
    y = jnp.sin(theta / 2) * jnp.sin(phi)
    z = jnp.cos(theta / 2)
    return jnp.array([x, y, z])


def cartesian_to_spherical(v: Array) -> Array:
    """Convert Cartesian unit vectors to spherical angles"""
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = jnp.sqrt(x**2 + y**2)
    theta = 2 * jnp.arctan2(r, z)
    phi = jnp.mod(jnp.arctan2(y, x), TWO_PI)
    return jnp.stack([theta, phi], axis=-1)


def random_VMF_angle(kappa: float, n: int, d: int = 3) -> np.ndarray:
    """Sample angular components from a von Mises-Fisher distribution."""
    if kappa <= 0:
        return np.random.uniform(-1.0, 1.0, size=n)

    alpha = (d - 1) / 2
    r0 = np.sqrt(1 + (alpha / kappa) ** 2) - alpha / kappa
    t0 = r0
    log_t0 = kappa * t0 + (d - 1) * np.log(1 - r0 * t0)

    found = 0
    out = []
    while found < n:
        batch = max(1, int((n - found) * 1.5))
        t = np.random.beta(alpha, alpha, batch)
        t = 2 * t - 1
        t = (r0 + t) / (1 + r0 * t)
        log_acc = kappa * t + (d - 1) * np.log(1 - r0 * t) - log_t0
        accepted = t[np.random.random(batch) < np.exp(log_acc)]
        out.append(accepted)
        found += len(accepted)

    return np.concatenate(out)[:n]


def random_VMF(mu: Array, kappa: float, size=None) -> np.ndarray:
    """Generate angle pairs concentrated around a reference direction."""
    n = 1 if size is None else int(np.prod(size))
    shape = () if size is None else tuple(np.ravel(size))
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape

    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    cos = random_VMF_angle(kappa, n, d=d)
    sin = np.sqrt(1 - cos**2)
    samples = z * sin[:, None] + cos[:, None] * mu[None, :]
    return samples.reshape((*shape, d))


def VMF_samples(theta0: float, phi0: float, kappa: float, num_samples: int) -> Array:
    """Initialize a mesh by sampling MZI parameters from a given distribution."""
    mu = spherical_to_cartesian(theta0, phi0)
    samples = random_VMF(mu, kappa, size=(num_samples,))
    return cartesian_to_spherical(samples)


def mesh_init_from_distribution( m: int,distribution: Callable,*distribution_params, ansatz: str):
    """Distribution-based initialization for the Butterfly ansatz"""
    if "num_samples" not in inspect.signature(distribution).parameters:
        raise TypeError(f"Distribution {distribution.__name__} must have a 'num_samples' parameter.")

    pattern_fn, unitary_fn = pattern_and_unitary(ansatz)
    num_mzi = pattern_fn(m).shape[0]
    params = distribution(*distribution_params, num_samples=num_mzi)
    gammas = jnp.zeros(m)
    return unitary_fn(params, gammas, m), params, gammas

# Distribution-based initialization for all the ansatz
def butterfly_init(m: int, distribution: Callable, *distribution_params):
    return mesh_init_from_distribution(m, distribution, *distribution_params, ansatz="butterfly")


def clements_init(m: int, distribution: Callable, *distribution_params):
    return mesh_init_from_distribution(m, distribution, *distribution_params, ansatz="clements")


def mzi3_init(m: int, distribution: Callable, *distribution_params):
    return mesh_init_from_distribution(m, distribution, *distribution_params, ansatz="mzi3")
