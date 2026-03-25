# External package imports
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy import sparse

# Internal imports from our QeMCMC package

# sampler contains the classical and quantum enhanced MCMC sampler
from qemcmc.sampler import ClassicalMCMC, QeMCMC
from qemcmc.sampler.runners import MCMCRunner, ConstrainedMCMCRunner
from qemcmc.sampler.runners import ConstrainedMCMCRunner


# These are object useful to store the results of the MCMC simulation
from qemcmc.utils import plot_chains

# This helps you build an example Ising model
from qemcmc.utils import ModelMaker
import itertools
from qemcmc.model import EnergyModel


# Define parameters
n = 20  # Number of spins in the system
coarse_graining_number = n
n_spins = n
reps = 5  # How many markov chains to produce
steps = 10000  # Length of each markov chain
temp = 0.1  # Temperature of the system


# QeMCMC parameters
gamma = (0.3, 0.6)  # Relative strength of mixer hamiltonian
time = (1, 20)  # Time for hamiltonian simulation

# Make all combinations of subgroups that include coarse_graining_number of integers between 0 and n_spins -1
# Feroz, could we add this simple functionality to ModelMaker?
subgroups = list(itertools.combinations(range(n_spins), coarse_graining_number))


shape_of_J = (n_spins, n_spins)
J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
J_tril = np.tril(J, -1)
J_triu = J_tril.transpose()
J = J_tril + J_triu

h = np.round(np.random.normal(0, 1, n_spins), decimals=4)

couplings = [h, J]
# Note that we could add a term representing the constraint to this, 
# but it would be highly innefficient in terms of hamiltonian simulation
# Instead we just leave it out.
#
# J_n = np.zeros((n_spins,)*n_spins)


# h_i*Z_i
# J_ij*Z_i*Z_j
# J_i...n*Z_i*Z_j*Z_k*...*Z_n


# # loop through the indices of J 
# for i,j,k in range(n_spins):
#     if i != j, i != k, j != k:
#         add term J_ijk*Z_i*Z_j*Z_k to hamiltonian

# J_sparse = {(0,1,2): 1.3, (0,3,4): -0.7, (1,2,4): 0.5} # example sparse representation of 3-local couplings
# J_paulis = [1.3*ZZZIII, -0.7*ZIIIZZ, 0.5*IZZIIZ] # example representation of the same couplings in terms of Pauli strings (using some hypothetical notation where ZZZIII means Z on qubits 0,1,2 and I on 3,4,5)
# # loop through the indices of J 
# for term in J_sparse:
#     add term.key * Z_term[0]*Z_term[0]*Z_term[0] to hamiltonian


# Define a constraint function: magnetization must be zero (equal number of 0s and 1s)
def zero_magnetization_constraint(bitstring: str) -> bool:
    # In {0, 1} representation, zero magnetization means sum of bits is n/2
    return bitstring.count('1') == len(bitstring) // 2

if n%2 != 0:
    raise ValueError("n must be even for zero magnetization constraint to be satisfiable")
model = EnergyModel(n=n_spins, couplings=couplings)
init_states = []
while len(init_states) < reps:
    state = model.get_random_state()
    if zero_magnetization_constraint(state):
        init_states.append(state)

model.initial_state = init_states
initial_states = model.initial_state
print("Initial states:", initial_states)




# Run classical (uniform and local) chains

runner = ConstrainedMCMCRunner(zero_magnetization_constraint)

uni_chains = []
for rep in range(reps):
    classical_uniform_MCMC = ClassicalMCMC(model, temp, method="uniform")
    uni_chain = runner.run(classical_uniform_MCMC, steps, initial_state=initial_states[rep], name="classical uniform MCMC", verbose=False, sample_frequency=1)
    uni_chains.append(uni_chain)


loc_chains = []
for rep in range(reps):
    classical_local_MCMC = ClassicalMCMC(model, temp, method="local")
    loc_chain = runner.run(classical_local_MCMC, steps, initial_state=initial_states[rep], name="classical local MCMC", verbose=False, sample_frequency=1)
    loc_chains.append(loc_chain)

loc2_chains = []
for rep in range(reps):
    classical_local_MCMC = ClassicalMCMC(model, temp, method="2-local")
    loc2_chain = runner.run(classical_local_MCMC, steps, initial_state=initial_states[rep], name="classical 2-local MCMC", verbose=False, sample_frequency=1)
    loc2_chains.append(loc2_chain)

# Run the quantum algorithm
# The QeMCMC class is very similar to the MCMC class, but it has a few extra parameters that are specific to the quantum algorithm.
# gamma and time are the hyperparameters that might need tuned for your specific problem.
# This may take a minute or so to run, depending on the number of steps and the size of the system. (takes 40s on my system for the default settings)


def run_qemcmc(rep):
    quantum_MCMC = QeMCMC(model, gamma=gamma, time=time, temp=temp)
    return runner.run(quantum_MCMC, steps, initial_state=initial_states[rep], name="QeMCMC", verbose=True, sample_frequency=1)


# Run in parallel as they can take a while.
Qe_chains = Parallel(n_jobs=-1)(delayed(run_qemcmc)(rep) for rep in range(reps))


# Plot the list of chains with a helper function
# Dark line is for average ove rchains, transparenr lines are for each individual chains

plot_chains(uni_chains, "orange", "classical uniform MCMC", plot_individual_chains=False)
plot_chains(loc_chains, "lightgreen", "classical local MCMC", plot_individual_chains=False)
plot_chains(loc2_chains, "darkgreen", "classical 2-local MCMC", plot_individual_chains=False)
plot_chains(Qe_chains, "lightblue", "QeMCMC", plot_individual_chains=False)

lowest_energies, degeneracies, lowest_configs = model.get_lowest_energies(10, return_configurations=True)

for i in range(len(lowest_energies)):
    if degeneracies[i] > 1:
        for j in range(degeneracies[i]):
            if zero_magnetization_constraint(lowest_configs[i][j]):
                plt.axhline(lowest_energies[i], color="red", linestyle="--", label=f"Ground state energy (degeneracy {degeneracies[i]})")
    else:
        if zero_magnetization_constraint(lowest_configs[i][0]):
            plt.axhline(lowest_energies[i], color="red", linestyle="--", label="Ground state energy")


plt.xlabel("MCMC step")
plt.ylabel("Energy")
plt.title("Classical vs Quantum-enhanced chains | T = {}".format(temp))
plt.legend()
plt.show()
