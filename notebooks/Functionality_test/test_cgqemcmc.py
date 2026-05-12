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
    return runner.run(**kwargs)

def run_visual_tests():
    # Setup standard proble
    n = 10
    reps = 5
    steps = 150
    np.random.seed(1)
    start_time = time.time()


    model = ModelMaker(n , model_type="Fully Connected Ising", name = "Ising").model
    
    # 1. Standard QeMCMC Test
    print("Running Standard QeMCMC...")
    proposer = QeProposal(
        model=model,
        gamma=(0.3, 0.6),
        time=(1, 20),
    )
    runner = MCMCRunner(model=model, temp=0.1)
    
    chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner, proposer=proposer, n_hops=steps, initial_state="0"*n, verbose=False)
        for seed in np.arange(0,reps))#np.random.randint(0, 2**31 - 1, size=reps)
    

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
        model=model,
        gamma=(0.3, 0.6),
        time=(1, 20),
        coarse_graining=cg
    )


    print("\nRunning Coarse-Grained manual QeMCMC...")
    cg_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner, proposer=cg_proposer, n_hops=steps, initial_state="0"*n, verbose=False)
        for seed in np.arange(0,reps))#np.random.randint(0, 2**31 - 1, size=reps)
    

    plot_chains(cg_chains, "lightblue", label = "Coarse-Grained manual QeMCMC", plot_individual_chains=True)

    cg_proposal_2 = QeProposal(
        model=model,
        gamma=(0.3, 0.6),
        time=(1, 20),
        m = 2,
    )
    print("\nRunning Coarse-Grained automatic QeMCMC...")
    cg_2_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner, proposer=cg_proposal_2, n_hops=steps, initial_state="0"*n, verbose=False)
        for seed in np.arange(0,reps))#np.random.randint(0, 2**31 - 1, size=reps)
    

    plot_chains(cg_2_chains , "blue", label = "Coarse-Grained automatic QeMCMC", plot_individual_chains=True)

    print("\nRunning Classical local MCMC...")
    # classical to compare against
    loc_proposer = ClassicalProposal(model, method = "local")
    loc_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner, proposer=loc_proposer, n_hops=steps, initial_state="0"*n, verbose=False)
        for seed in np.arange(0,reps))#np.random.randint(0, 2**31 - 1, size=reps)
    

    plot_chains(loc_chains, "green", label = "Classical local MCMC", plot_individual_chains=True)
    print("\nRunning Classical uniform MCMC...")
    uni_proposer = ClassicalProposal(model, method = "uniform")
    uni_chains = Parallel(n_jobs=-1)(
        delayed(run_chain_with_seed)(seed, runner, proposer=uni_proposer, n_hops=steps, initial_state="0"*n, verbose=False)
        for seed in np.arange(0,reps))#np.random.randint(0, 2**31 - 1, size=reps)
    

    plot_chains(uni_chains, "orange", label = "Classical uniform MCMC", plot_individual_chains=True)


    
    plt.title("QeMCMC thermalisation")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    

    end_time = time.time()

    # As this is a test, add some baseline time taken
    plt.figtext(0.15, 0.85, f"Time taken: {end_time - start_time:.2f} seconds", fontsize=10)

    plt.legend()
    #add time signature to save file
    plt.savefig(os.path.join(plot_dir, f"test_cgqemcmc_{int(time.time())}.png"))
    plt.close()

    print(f"\nVisual tests complete. Plots saved to: {os.path.abspath(plot_dir)}")

if __name__ == "__main__":
    run_visual_tests()