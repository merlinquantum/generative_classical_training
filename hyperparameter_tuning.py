import jax.numpy as jnp
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
from src.helpers.initialization import random_clements_init, close_to_identity_clements_init
from src.helpers.initialization import random_mzi3_init, close_to_identity_mzi3_init
from src.helpers.initialization import random_butterfly_init, close_to_identity_butterfly_init
from src.helpers.initialization import random_haar_init, close_to_identity_haar_init
from src.models.mmd_estimator import MMD_loss, MMD_loss_butterfly, MMD_loss_mzi3, MMD_loss_haar
from src.models.training import Trainer
from src.helpers.utils import median_heuristic, pack_params, generate_init_state
from ray import tune
from ray.tune.logger import TBXLogger
import os
import traceback
from datetime import datetime

def train_fn(config):

    # Pick m and n
    m = config['m']
    n = config['n']

    # Parameters common to more than one model
    n_iters = config['n_iters']
    sigma_choice = config['sigma']

    # Load dataset
    path_user = ''
    train_path = path_user + 'src/data/preference_ranking/sushi_train.csv'
    test_path = path_user + 'src/data/preference_ranking/sushi_test.csv'
    X_train = jnp.array(np.loadtxt(train_path, delimiter = ','))
    X_test = jnp.array(np.loadtxt(test_path, delimiter = ','))
    print('Data imported')

    # If sigma is not a fixed number
    if sigma_choice == 'median_heuristic':
        sigma = jnp.array(jnp.sqrt(median_heuristic(np.array(X_train))))
    elif sigma_choice == 'root_4':
        sigma = jnp.array(m**(1/4))
    else:
        sigma = jnp.array(sigma_choice)
       
    print('Sigma:' + str(sigma))
    
    # Set up keys
    ts = int(datetime.now().timestamp())
    key0 = jax.random.PRNGKey(ts)
    key, start_key = jax.random.split(key0, 2)
    key2, init_key = jax.random.split(key, 2)
    key3, mmd_key = jax.random.split(key2, 2)
    key4, test_key = jax.random.split(key3, 2)

    # Get config parameters
    optimizer = config['optimizer']
    stepsize = config['stepsize']
    n_samples_operators = config['n_samples_operators']
    n_samples_gurvits = config['n_samples_gurvits']
    initialization_strategy = config['initialization_strategy']
    ansatz = config['ansatz']
    init_state_type = config['init_state_type']
    perturbation = config['perturbation']
    
    # Get loss function
    if ansatz == "clements":
        Loss = MMD_loss
    elif ansatz == "butterfly":
        Loss = MMD_loss_butterfly
    elif ansatz == "mzi3":
        Loss = MMD_loss_mzi3
    elif ansatz == 'haar':
        Loss = MMD_loss_haar
    else:
        print('Ansatz not available')

    # Get initialization
    if isinstance(initialization_strategy, str):
        if initialization_strategy == 'random':
            if ansatz == 'clements':
                U_init, params_mzi_init, gammas_init = random_clements_init(m, init_key)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'mzi3':
                U_init, params_mzi_init, gammas_init = random_mzi3_init(m, init_key)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'butterfly':
                U_init, params_mzi_init, gammas_init = random_butterfly_init(m, init_key)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'haar':
                U_init, params_init = random_haar_init(m, init_key)
        elif initialization_strategy == 'close_to_identity':
            if ansatz == 'clements':
                U_init, params_mzi_init, gammas_init = close_to_identity_clements_init(m,
                                                                                       init_key,
                                                                                       max_value_theta = perturbation,
                                                                                       max_value_phi = perturbation,
                                                                                       max_value_gamma = perturbation)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'mzi3':
                U_init, params_mzi_init, gammas_init = close_to_identity_mzi3_init(m,
                                                                                   init_key,
                                                                                   max_value_theta = perturbation,
                                                                                   max_value_phi = perturbation,
                                                                                   max_value_gamma = perturbation)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'butterfly':
                U_init, params_mzi_init, gammas_init = close_to_identity_butterfly_init(m,
                                                                                        init_key,
                                                                                        max_value_theta = perturbation,
                                                                                        max_value_phi = perturbation,
                                                                                        max_value_gamma = perturbation)
                params_init = pack_params(params_mzi_init, gammas_init)
            elif ansatz == 'haar':
                U_init, params_init = close_to_identity_haar_init(m, init_key, max_value_perturb = perturbation)
            
    elif isinstance(initialization_strategy, np.ndarray): 
        params_init = jnp.array(initialization_strategy)
    else:
        print('Initialization not available')

    # Generate input state
    init_state = generate_init_state(m = m, n = n, init_state_type = init_state_type)
    init_state_ind = jnp.where(init_state)[0]

    # Loss function arguments
    loss_kwargs = {
        "params": params_init,
        "target_dataset": X_train,
        "sigma": sigma,
        "m": m,
        "n": n,
        "key": mmd_key,
        "n_samples_operators": n_samples_operators,
        "n_samples_gurvits": n_samples_gurvits,
        "init_state_ind": init_state_ind
    }

    # Initialize Trainer class
    trainer = Trainer(optimizer = optimizer, loss = Loss, stepsize = stepsize, opt_jit=False)

    # Optimize
    try:
        # Launch trainer
        print('Start training')
        trainer.train(n_iters, loss_kwargs, val_kwargs=None,
                      monitor_interval=None, turbo=None,
                      convergence_interval=200)

        # Evaluate MMD on test set
        print('Get test loss')
        test_loss = Loss(circuit_parameters = trainer.final_params,
                         target_dataset = X_test,
                         sigma = sigma,
                         m = m,
                         n = n,
                         key = test_key,
                         n_samples_operators = n_samples_operators,
                         n_samples_gurvits = n_samples_gurvits,
                         init_state_ind = init_state_ind)
        
        # Save parameters
        trial_dir = tune.get_context().get_trial_dir()
        final_params_path = os.path.join(trial_dir, "final_parameters.npy")
        np.save(final_params_path, np.asarray(trainer.final_params))

        losses_path = os.path.join(trial_dir, "losses.npy")
        np.save(losses_path, np.asarray(trainer.losses))
        
        # Save results to report
        tune.report({"final_loss": trainer.losses[-1],
                     "final_params_path": final_params_path, 
                     "test_loss": test_loss,
                     "losses_path": losses_path,
                     "runtime": trainer.run_time})
        
        return
    
    except Exception as e:
        print("Trial failed with exception:", repr(e))
        traceback.print_exc()
        raise

        

# If loading init parameters from another training
#warm_start_path = ''
#initialization_strategy = np.load(warm_start_path)

# QCBM search space example
search_space_qcbm = {
    "m": 100,
    "n": 10,
    "ansatz": tune.grid_search(['clements', 'haar', 'butterfly', 'mzi3']), 
    "n_iters": 2000,
    "sigma": 3.0,
    "optimizer": "Adam",
    "stepsize": 0.01,
    "n_samples_operators": 2000,
    "n_samples_gurvits": 5000, 
    "init_state_type": 'middle_alternating',
    "initialization_strategy": "close_to_identity",
    "perturbation": 0.5
}


if __name__ == "__main__":
    # num_samples defines how many times a given configuration is repeated
    num_samples = 1
    
    # run optimization
    analysis = tune.run(train_fn, config=search_space_qcbm, num_samples = num_samples, raise_on_failed_trial = False)
