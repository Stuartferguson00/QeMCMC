
# Import necessary libraries
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# Import the required classes from QeMCMC

# QeMCMC class contains the main functionality for the classical MCMC simulation
from qemcmc.ClassicalMCMC import ClassicalMCMC

# QeMCMC class contains the main functionality for the quantum enhanced MCMC simulation
from qemcmc.QeMCMC_ import QeMCMC

# These are object useful to store the results of the MCMC simulation
from qemcmc.helpers import plot_chains

# This helps you build an example Ising model
from qemcmc.ModelMaker import ModelMaker


# Define parameters
n = 10 # Number of spins in the system
reps = 5 # How many markov chains to produce
steps = 30 # Length of each markov chain
temp = 0.1 # Temperature of the system


# QeMCMC parameters
gamma = (0.3,0.6)# Relative strength of mixer hamiltonian 
time = (1,20) # Time for hamiltonian simulation

# Build an Ising model to test the algorithms on
model_type = "Coarse Grained Ising"
name = "Test Ising model"
model = ModelMaker(n, model_type, name).model
initial_states = model.initial_state




# Run classical (uniform and local) chains

uni_chains = []
for rep in range(reps):
    classical_uniform_MCMC = ClassicalMCMC(model, temp, method = "uniform")
    uni_chain = classical_uniform_MCMC.run(steps, initial_state = initial_states[rep], name = "classical uniform MCMC", verbose = False, sample_frequency = 1)
    uni_chains.append(uni_chain)


loc_chains = []
for rep in range(reps):
    classical_local_MCMC = ClassicalMCMC(model, temp, method = "local")
    loc_chain = classical_local_MCMC.run(steps, initial_state = initial_states[rep], name = "classical local MCMC", verbose = False, sample_frequency = 1)
    loc_chains.append(loc_chain)


# Run the quantum algorithm
# The QeMCMC class is very similar to the MCMC class, but it has a few extra parameters that are specific to the quantum algorithm. 
# gamma and time are the hyperparameters that might need tuned for your specific problem.
# This may take a minute or so to run, depending on the number of steps and the size of the system. (takes 40s on my system for the default settings)

def run_qemcmc(rep):
    quantum_MCMC = QeMCMC(model, gamma=gamma, time=time, temp=temp)
    return quantum_MCMC.run(steps, initial_state=initial_states[rep], name="QeMCMC", verbose=True, sample_frequency=1)

# Run in parallel as they can take a while.
Qe_chains = Parallel(n_jobs=-1)(delayed(run_qemcmc)(rep) for rep in range(reps))



# Plot the list of chains with a helper function
# Dark line is for average ove rchains, transparenr lines are for each individual chains

plot_chains(uni_chains, "orange", "classical uniform MCMC")
plot_chains(loc_chains, "lightgreen", "classical local MCMC")
plot_chains(Qe_chains, "lightblue", "QeMCMC")


plt.xlabel("MCMC step")
plt.ylabel("Energy")
plt.title("Classical vs Quantum-enhanced chains | T = {}".format(temp))
plt.legend()
plt.show()