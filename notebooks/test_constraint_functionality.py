
import os

import os
from time import time

import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy import sparse
import itertools

# Internal imports from our QeMCMC package
from qemcmc.sampler import ClassicalProposal, QeProposal
from qemcmc.sampler.runners import MCMCRunner, ConstrainedMCMCRunner
from qemcmc.utils import plot_chains, get_random_state, ModelMaker
from qemcmc.model import EnergyModel, ConstraintModel

n = 10  # Number of spins in the system
n_spins = n
reps = 6  # How many markov chains to produce
temp = 0.1  # Temperature of the system

# Make baseline Ising gmodel
model = ModelMaker(n, "Fully Connected Ising", "Base Ising Model").model
shape_of_J = (n_spins, n_spins)
J_constraint = np.zeros(shape_of_J)
#J_constraint[-2, -1] = 1
#J_constraint[-1, -2] = 1
for i in range(n_spins):
    for j in range(i+1, n_spins):
        u = np.random.uniform(0, 1)
        if u < 0.05:# 10% chance of adding a constraint between these spins
            J_constraint[i, j] = 1
            J_constraint[j, i] = 1
# To ensure non-constant constraint
J_constraint[0,1] = 1
J_constraint[1,0] = 1
print("Constraint Couplings:\n", J_constraint)


def constraint_checker_func(bitstring: str) -> bool:
    """In {0, 1} representation, this constraint means no two adjacent bits can both be 1"""

    sum = 0
    for i in range(len(bitstring)):
        for j in range(i+1, len(bitstring)):
            if bitstring[i] != bitstring[j]:
                sum += J_constraint[i, j]

    
    return sum == 0

constraint_model = ConstraintModel(n, constraint_couplings = [J_constraint,], couplings = model.couplings, name="Constraint Model", cost_function_signs=model.cost_function_signs, constraint_func=constraint_checker_func, constraint_signs = [-1,])

Runner = ConstrainedMCMCRunner(constraint_model, temp=temp, reject_invalid=True)


def run_constrained_chain(method, rep, coupling_weights=None):
    if method == "uniform":
        proposer = ClassicalProposal(constraint_model, method="uniform")
        name = "Constrained Classical Uniform MCMC"
        steps = 10000
    elif method == "local":
        proposer = ClassicalProposal(constraint_model, method="local")
        name = "Constrained Classical Local MCMC"
        steps = 10000
    elif method == "quantum":
        if coupling_weights is None:
            coupling_weights = [1,1,1]
        proposer = QeProposal(constraint_model, gamma=0.5, time=10, m=1, coupling_weights=coupling_weights)
        name = "Constrained QeMCMC"
        steps = 100
    # Runner.run returns (chain, rejection_count)
    chain, rejection_count = Runner.run(proposer=proposer, n_hops=steps, initial_state=constraint_model.initial_state[rep], name=name, verbose=True, sample_frequency=1)
    print(f"Coupling weights: {coupling_weights}, Rejection count: {rejection_count}")
    rejection_rate = rejection_count / steps
    return chain, rejection_rate

"""uni_chains, _ = Parallel(n_jobs=-1)(delayed(run_constrained_chain)("uniform", rep) for rep in range(reps))
loc_chains, _ = Parallel(n_jobs=-1)(delayed(run_constrained_chain)("local", rep) for rep in range(reps))
qe_chains, _ = Parallel(n_jobs=-1)(delayed(run_constrained_chain)("quantum", rep) for rep in range(reps))


plot_chains(uni_chains, "orange", "classical uniform MCMC", plot_individual_chains=False)
plot_chains(loc_chains, "lightgreen", "classical local MCMC", plot_individual_chains=False)
plot_chains(qe_chains, "lightblue", "Qe energy", plot_individual_chains=False)
plt.show()"""

# Plot the releationship between the rejection count vs the coupling weights for the problem hamiltonian and constraint hamiltonian
w_ab_range = np.logspace(-1, 0, 20)
w_c_range = np.logspace(-1, 1, 20)
print("w_ab_range: ", w_ab_range)
print("w_c_range: ", w_c_range)
coupling_weights_list = [[w_ab, w_ab, w_c] for w_ab in w_ab_range for w_c in w_c_range]

start = time()
output = Parallel(n_jobs=-1)(delayed(run_constrained_chain)("quantum", 0, coupling_weights=coupling_weights_list[rep]) for rep in range(len(coupling_weights_list)))
end = time()
print(f"Total time for running all chains: {end - start:.2f} seconds")
output = np.array(output, dtype=object)
qe_chains_both = output[:, 0]
# Reshape the rejection counts into a 2D grid for plotting
Qe_rejection = output[:, 1].reshape((len(w_ab_range), len(w_c_range))).astype(float)
W_ab, W_c = np.meshgrid(w_ab_range, w_c_range, indexing='ij')
# Plot the results
print("Qe_rejection")
plt.figure()
#plt.imshow(Qe_rejection, origin='lower', extent=[w_c_range[0], w_c_range[-1], w_ab_range[0], w_ab_range[-1]], aspect='auto', interpolation='nearest')
plt.scatter(W_c.flatten(), W_ab.flatten(), c=Qe_rejection.flatten(), cmap='viridis', marker='s')
plt.xlabel("w_c")
plt.ylabel("w_ab")
plt.colorbar(label="Rejection Rate")
plt.title("Rejection Rate vs Coupling Weights")
plt.yscale('log')
plt.xscale('log')
plt.show()




