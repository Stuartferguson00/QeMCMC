import itertools
import time
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
import os

from qemcmc.model import EnergyModel, ConstraintModel, ModelMaker
from qemcmc.coarse_grain import CoarseGraining
from qemcmc.sampler.classical_proposal import ClassicalProposal
from qemcmc.sampler.qe_proposal import QeProposal
from qemcmc.sampler.runners import MCMCRunner, ConstrainedMCMCRunner
from qemcmc.utils import plot_chains

# Setup directory for plots
# get file directiory
path_ = os.path.dirname(__file__)

plot_dir = os.path.join(path_, "test_results_plots")

# make sure relative path exists
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def get_states(chain_obj):
    """Helper to extract states from an MCMCChain object."""
    if hasattr(chain_obj, 'states'):
        return chain_obj.states
    return list(chain_obj)

def run_chain_with_seed(seed, runner, **kwargs):
    np.random.seed(seed)
    return runner.run(**kwargs, return_rejections=False)

def run_visual_tests():
    start_time = time.time()
    # Setup standard proble
    n = 8
    reps = 7
    steps = 150
    temp = 0.1
    np.random.seed(5)


    base_model = ModelMaker(n , model_type="Fully Connected QUBO", name = "Binary Ising").model
    


    shape_of_J = (n, n)
    J_constraint = np.zeros(shape_of_J)
    for i in range(n):
        for j in range(i+1, n):
            u = np.random.uniform(0, 1)
            if u < 0.05:# 10% chance of adding a constraint between these spins
                J_constraint[i, j] = 1
                J_constraint[j, i] = 1


    def constraint_checker_func(bitstring: str) -> bool:
        """In {0, 1} representation, this constraint means no two adjacent bits can both be 1"""

        sum = 0
        for i in range(len(bitstring)):
            for j in range(i+1, len(bitstring)):
                if bitstring[i] != bitstring[j]:
                    sum += J_constraint[i, j]
        return sum == 0

    model_constraint = ConstraintModel(n, constraint_couplings = [J_constraint,], couplings = base_model.couplings, name="Large Constraint Model", cost_function_signs=base_model.cost_function_signs, constraint_func=constraint_checker_func, constraint_signs = [-1,], model_type="binary")  
    random_initial_states = model_constraint.initial_state
    runner_constraint = ConstrainedMCMCRunner(model_constraint, temp, constraint_checker_func)

    # 1. Standard QeMCMC Test
    print("Running Standard QeMCMC...")
    proposer = QeProposal(
        model=model_constraint,
        gamma=(0.1, 0.4),
        time=(1, 20),
        coupling_weights=[0.1, 0.1, 0.7],  
    )
    
    chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner_constraint, proposer=proposer, n_hops=steps, initial_state=random_initial_states[iter], verbose=False)
        for iter, seed in enumerate(np.arange(0,reps)))
    

    plot_chains(chains, "red", label = "Standard QeMCMC", plot_individual_chains=True)


    subgroups = list(itertools.combinations(range(n), n//2))
    len_subgroups = len(subgroups)
    # 3. Coarse-Grained QeMCMC Test
    cg = CoarseGraining(
        n=n,
        subgroups=subgroups,
        subgroup_probs=np.ones(len_subgroups)/len_subgroups
    )

    cg_proposer = QeProposal(
        model=model_constraint,
        gamma=(0.1, 0.4),
        time=(1, 20),
        coarse_graining=cg,
        coupling_weights=[0.1, 0.1, 0.7],  

    )


    print("\nRunning Coarse-Grained manual QeMCMC...")
    cg_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner_constraint, proposer=cg_proposer, n_hops=steps, initial_state=random_initial_states[iter], verbose=False)
        for iter, seed in enumerate(np.arange(0,reps)))
    

    plot_chains(cg_chains, "lightblue", label = "Coarse-Grained manual QeMCMC", plot_individual_chains=True)

    cg_proposal_2 = QeProposal(
        model=model_constraint,
        gamma=(0.1, 0.4),
        time=(1, 20),
        m = 2,
        coupling_weights=[0.1, 0.1, 0.7],  
    )
    print("\nRunning Coarse-Grained automatic QeMCMC...")
    cg_2_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner_constraint, proposer=cg_proposal_2, n_hops=steps, initial_state=random_initial_states[iter], verbose=False)
        for iter, seed in enumerate(np.arange(0,reps)))
    

    plot_chains(cg_2_chains , "blue", label = "Coarse-Grained automatic QeMCMC", plot_individual_chains=True)

    print("\nRunning Classical local MCMC...")
    # classical to compare against
    loc_proposer = ClassicalProposal(model_constraint, method = "local")
    loc_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner_constraint, proposer=loc_proposer, n_hops=steps, initial_state=random_initial_states[iter], verbose=False)
        for iter, seed in enumerate(np.arange(0,reps)))
    

    plot_chains(loc_chains, "green", label = "Classical local MCMC", plot_individual_chains=True)
    print("\nRunning Classical uniform MCMC...")
    uni_proposer = ClassicalProposal(model_constraint, method = "uniform")
    uni_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner_constraint, proposer=uni_proposer, n_hops=steps, initial_state=random_initial_states[iter], verbose=False)
        for iter, seed in enumerate(np.arange(0,reps)))
    

    plot_chains(uni_chains, "orange", label = "Classical uniform MCMC", plot_individual_chains=True)

    end_time = time.time()
    
    plt.title("QeMCMC thermalisation")
    plt.xlabel("Step")
    plt.ylabel("Energy")

    # As this is a test, add some baseline time taken
    plt.figtext(0.15, 0.85, f"Time taken: {end_time - start_time:.2f} seconds", fontsize=10)

    plt.legend()

    #add time signature to save file
    plt.savefig(os.path.join(plot_dir, f"test_constrained_{int(time.time())}.png"))
    plt.close()

    print(f"\nVisual tests complete. Plots saved to: {os.path.abspath(plot_dir)}")

if __name__ == "__main__":
    run_visual_tests()