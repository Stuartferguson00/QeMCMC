import argparse
import sys
import json
import os
import pickle
import datetime
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy import sparse
import itertools
import matplotlib.pyplot as plt

# Internal imports from our QeMCMC package
from qemcmc.coarse_grain import CoarseGraining
from qemcmc.sampler import ClassicalProposal, QeProposal
from CST_classical_proposals import CSTClassicalProposal
from qemcmc.sampler.runners import MCMCRunner, ConstrainedMCMCRunner
from qemcmc.utils import plot_chains, get_random_state
from qemcmc.model import EnergyModel, ConstraintModel, ModelMaker, constraint_model
from CST_helpers import *
from tabulate import tabulate
import time

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    C = config.get("C")
    temp = config.get("temp")
    epsilon = config.get("epsilon")
    uniform = config.get("uniform")
    
    n = C * (C - 1) // 2 # Number of (Qu)bits needed to represent the spacetime

    print(f"Using {n} bits to represent a causal set of cardinality {C}.")
    print("Runnign experiment with the following parameters:")
    # use tabulate to print parameters in a nice table format
    params_table = [["C", C], ["temp", temp], ["epsilon", epsilon], ["uniform", uniform]]
    print(tabulate(params_table, headers=["Parameter", "Value"], tablefmt="grid"))
    # And do the same for the inidivdual expeirments
    for exp in config.get("experiments", []):
        exp_name = exp.get("name", "experiment")
        exp_type = exp.get("type", "classical")
        exp_params = exp.get("params", {})
        print(f"\nExperiment: {exp_name} (Type: {exp_type})")
        exp_params_table = [[k, v] for k, v in exp_params.items()]
        print(tabulate(exp_params_table, headers=["Parameter", "Value"], tablefmt="grid"))


    BD_couplings = get_BD_couplings_4d(C, epsilon=epsilon)
    BD_couplings_list = [BD_couplings[0], BD_couplings[1], BD_couplings[2]]

    BD_action_model = EnergyModel(n, couplings = BD_couplings_list, name = "BD Action Model", cost_function_signs =[1,1,1], model_type = "binary")

    def constraint_checker_func(bitstring: str) -> bool:
        return is_causal_matrix(bitstring_to_matrix(bitstring))

    TC_couplings = get_TC_couplings(C)
    TC_couplings_list = [TC_couplings[0], TC_couplings[1]]
    TC_model = ConstraintModel(n, constraint_couplings = [coupling for coupling in TC_couplings_list], name="TC Constraint Model", constraint_func=constraint_checker_func, constraint_signs = [1 for coupling in TC_couplings_list], couplings = None, model_type ="binary", cost_function_signs=None)

    initial_states_config = config.get("initial_states", ["0", "1"])
    initial_states = []
    for s in initial_states_config:
        if s == "0":
            initial_states.append("0" * n)
        elif s == "1":
            initial_states.append("1" * n)
        else:
            initial_states.append(s)

    BD_TC_model = ConstraintModel(n, constraint_couplings = [coupling for coupling in TC_couplings_list], name="TC Constraint Model", constraint_func=constraint_checker_func, constraint_signs = [1 for coupling in TC_couplings_list], couplings = [coupling for coupling in BD_couplings_list], model_type ="binary", cost_function_signs=[1 for coupling in BD_couplings_list])
    Runner = ConstrainedMCMCRunner(BD_TC_model, temp=temp, reject_invalid=True, uniform=uniform)

    num_steps_q = config.get("num_steps_q")
    num_steps_c = config.get("num_steps_c")

    def run_single_chain(exp, initial_state, chain_idx):
        exp_type = exp.get("type", "classical")
        params = exp.get("params", {})
        start_time = time.time()
        
        print(f"Running chain {chain_idx} for experiment '{exp.get('name')}' with initial state: {initial_state[:50]}...")  # Print the first 50 bits for brevity
        if exp_type == "qe":
            cg = CoarseGraining(n, repeated=params.get("repeated", False))
            try:
                cg_time = tuple(params.get("time"))
            except:
                cg_time = float(params.get("time", 1))
            print("cg_time: ", cg_time)

            coupling_weights = [1.0/(1-params.get("gamma")) for _ in range(len(TC_couplings_list))] 
            proposal = QeProposal(TC_model, gamma=params.get("gamma"), time=cg_time, m=params.get("m"), coarse_graining=cg,  coupling_weights=coupling_weights)
            res = Runner.run(proposal, n_hops=num_steps_q, initial_state=initial_state, verbose=True)
        elif exp_type == "cst":
            proposal = CSTClassicalProposal(TC_model, method=params.get("method", "link"))
            res = Runner.run(proposal, n_hops=num_steps_c, initial_state=initial_state, verbose=True)
        elif exp_type == "classical":
            proposal = ClassicalProposal(TC_model, method=params.get("method", "local"))
            res = Runner.run(proposal, n_hops=num_steps_c, initial_state=initial_state, verbose=True)
        else:
            return None
        tme = time.time() - start_time
        print(f"Finished chain {chain_idx} for experiment '{exp.get('name')}', took {tme} seconds.")
        return (exp.get("name"), chain_idx, res)
    
    # Flatten all tasks into a single list for global parallelization
    tasks = []
    for exp in config.get("experiments", []):
        for i, s in enumerate(initial_states):
            tasks.append((exp, s, i))

    print(f"\nRunning {len(tasks)} chains in parallel across all experiments...")
    raw_results = Parallel(n_jobs=-1)(
        delayed(run_single_chain)(*task) for task in tasks
    )
    
    # Reconstruct the results dictionary
    results = {}
    for exp in config.get("experiments", []):
        name = exp.get("name")
        print("raw_results:", raw_results)
        exp_chains = [r for r in raw_results if r is not None and r[0] == name]
        print(f"Collected results for experiment '{name}': {len(exp_chains)} chains.")
        print("exp_chains:", exp_chains)
        results[name] = {
            'params': exp.get("params", {}),
            'results': {f'chain_{r[1]}': r[2][0] for r in exp_chains},
            'constraint rejections': {f'rejections_{r[1]}': r[2][1] for r in exp_chains},
            'self rejections': {f'self_rejections_{r[1]}': r[2][2] for r in exp_chains},
            'MH rejections': {f'MH_rejections_{r[1]}': r[2][3] for r in exp_chains},
        }

    # Save results systematically
    save_dir = os.path.join(os.path.dirname(__file__), "saved_chains")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, f"experiments_{C}C_{temp}T_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    save_path = os.path.join(exp_dir, "data.pkl")

    global_params = {
        'C': C,
        'n': n,
        'temp': temp,
        'epsilon': epsilon,
        'num_steps_q': num_steps_q,
        'num_steps_c': num_steps_c,
        'initial_states': initial_states
    }

    save_data = {
        'global_params': global_params,
        'experiments': results
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nExperiments saved successfully to {save_path}\n")

    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spacetime Sampling Experiments")
    parser.add_argument("--config", type=str, default= os.path.join(os.path.dirname(__file__), "experiment_config.json"), help="Path to the JSON configuration file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found. Please create it or provide a valid path.")
        sys.exit(1)
        
    main(args.config)















"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)):
        calculated_BD = BD_action_model.get_energy(bitstring)
        expected_BD = calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False)
        print(f"Bitstring: {bitstring}, Calculated BD: {calculated_BD}, Expected BD: {expected_BD}")
        if not np.isclose(calculated_BD, expected_BD, atol=1e-10):
            print("Error: Calculated BD action does not match expected value!")
            print("as it is just first order approximation, this is fine...")
            # print(calculated_BD - expected_BD)
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False))
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=True, first_order_taylor=False))
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=True))
"""



"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and TC_model.get_constraint_energy(bitstring) == 0:
        pass
    if not is_causal_matrix(bitstring_to_matrix(bitstring)) and TC_model.get_constraint_energy(bitstring) <0.99:
        print("Error: Invalid state has low constraint energy!")
        print(bitstring)
    print(f"Bitstring: {bitstring}, Constraint Energy: {TC_model.get_constraint_energy(bitstring)}, Is Causal: {is_causal_matrix(bitstring_to_matrix(bitstring))}")
"""

"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and BD_TC_model.get_constraint_energy(bitstring) == 0:
        pass
    if not is_causal_matrix(bitstring_to_matrix(bitstring)) and BD_TC_model.get_constraint_energy(bitstring) <0.99:
        print("Error: Invalid state has low constraint energy!")
        print(bitstring)
    
    expected_BD = calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False)
    calculated_BD = BD_TC_model.get_energy(bitstring)
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and not np.isclose(calculated_BD, expected_BD, atol=1e-10):
        print("Error: Calculated BD action does not match expected value for valid state!")
        print(f"Bitstring: {bitstring}, Calculated BD: {calculated_BD}, Expected BD: {expected_BD}")

"""