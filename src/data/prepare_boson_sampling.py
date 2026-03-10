import perceval as pcvl
import numpy as np
import jax.numpy as jnp
from ..helpers.utils import generate_init_state


if __name__ == "__main__":
    m = 256
    n = 16

    # Define unitary, circuit, and input state in Perceval
    U_pcvl = pcvl.Matrix.random_unitary(m) 
    circ = pcvl.components.Unitary(U_pcvl)
    init_state = generate_init_state(m = m, n = n, init_state_type = 'middle_compact')
    init_state_pcvl = pcvl.BasicState(list(init_state))

    # Generate samples with Clifford backend
    backend = pcvl.Clifford2017Backend()
    backend.set_circuit(circ)
    backend.set_input_state(init_state_pcvl)
    clifford_samples = backend.samples(10000)

    # Reformat
    samples_clifford_list = []
    for sample in clifford_samples:
        samples_clifford_list.append(list(sample))
    
    samples_clifford_array = np.array(samples_clifford_list)

    # Check
    np.all(samples_clifford_array.sum(axis=1) == n)

    # Shuffle
    np.random.shuffle(samples_clifford_array)

    # Split in train /test
    X_train = samples_clifford_array[:5000]
    X_test = samples_clifford_array[5000:10000]

    # save
    path = './boson_sampling/'
    np.savetxt(path + 'boson_sampling_U1_train_m' + str(m) + '_n' + str(n) + '.csv', X_train, delimiter=',',  fmt='%d')
    np.savetxt(path + 'boson_sampling_U1_test_m' + str(m) + '_n' + str(n) + '.csv', X_test, delimiter=',', fmt='%d')
    np.save(path + 'U1.npy', np.asarray(U_pcvl))
    