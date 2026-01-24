import itertools
import matplotlib.pyplot as plt
from qemcmc.energy_models import EnergyModel
from qemcmc.CircuitMaker import CircuitMaker
from qemcmc.ModelMaker import ModelMaker
from qemcmc.QeMCMC_ import QeMCMC
import numpy as np
import tqdm
"""
n_spins = 8
m_q = 4
CG_sample_number = 2
brute_sample_multiplier= 1

reps = 1
get_data = True
gamma = (0.25,0.6)
time = (2,20)


temp = 1

model_dir = 'C:/Users/Stuart Ferguson/OneDrive - University of Edinburgh/Documents/PhD/CODE/CGQeMCMC/results/models_2/000.obj'
results_dir = 'C:/Users/Stuart Ferguson/OneDrive - University of Edinburgh/Documents/PhD/CODE/CGQeMCMC/results/hE_003/00_000.obj'

#change file names for easy file organisation
l_model_dir = list(model_dir)
if n_spins >=100:
    l_model_dir[-5] = str(n_spins)[2]
    l_model_dir[-6] = str(n_spins)[1]
    l_model_dir[-7] = str(n_spins)[0]
elif n_spins >=10:
    l_model_dir[-5] = str(n_spins)[1]
    l_model_dir[-6] = str(n_spins)[0]
else:
    l_model_dir[-5] = str(n_spins)[0]

model_dir = ''.join(l_model_dir)
"""

# Define parameters
n_spins = 9 # Number of spins in the system
reps = 1
coarse_graining_number = 9
# QeMCMC parameters
gamma = 0.6#(0.3,0.6)# Relative strength of mixer hamiltonian 
time = 10#(1,20) # Time for hamiltonian simulation

"""# Build an Ising model to test the algorithms on
model_type = "Coarse Grained Ising"
name = "Test Ising model"
model = ModelMaker(n_spins, model_type, name).model
"""

# Make all combinations of subgroups that include coarse_graining_number of integers between 0 and n_spins -1
# Feroz, could we add this simple functionality to ModelMaker?
subgroups = list(itertools.combinations(range(n_spins), coarse_graining_number))
print("subgroups:", subgroups)






shape_of_J = (n_spins, n_spins)
J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
J_tril = np.tril(J, -1)
J_triu = J_tril.transpose()
J = J_tril + J_triu

h = np.round(np.random.normal(0, 1, n_spins), decimals=4)

couplings = [h, J]
# why does the user have to calculate their own alpha? At least we should do it in model maker. Ask the user to input max_number of qubits, and we enumerate all combinations internally.
alpha = np.sqrt(n_spins) / np.sqrt(sum([J[i][j] ** 2 for i in range(n_spins) for j in range(i)]) + sum([h[j] ** 2 for j in range(n_spins)]))

model = EnergyModel(n=n_spins,
                   couplings=couplings,
                    subgroups=subgroups,
                    subgroup_probs=np.ones(len(subgroups))/len(subgroups),
                    alpha=alpha)
        
        




ell_enegries = model.get_all_energies()
S = np.array([np.binary_repr(s, width=n_spins) for s in range(2**n_spins)])
print("S:", S)
S_integer_arr = np.array([[int(bit) for bit in state] for state in S])
print("S_integer_arr:", S_integer_arr)
E_s = np.array(ell_enegries)
print("E_s:", E_s)
        
# find matrices of Eneegry differences of the configyurations, and hamming distances
# not efficient
E_diffs = np.zeros((2**n_spins,2**n_spins))
hamming_dists = np.zeros((2**n_spins,2**n_spins))
for s in S:
    for s_prime in S:


        hamming_dist = np.sum(np.bitwise_xor(S_integer_arr[int(s,2)],S_integer_arr[int(s_prime,2)]))
        E_diff = abs(E_s[int(s,2)]-E_s[int(s_prime,2)])
        E_diffs[int(s,2),int(s_prime,2)] = E_diff
        hamming_dists[int(s,2),int(s_prime,2)] = hamming_dist
print("hamming_dists:", hamming_dists)
print("E_diffs:", E_diffs)
            

# Weights are the probability of proposal from s to s'
local_weights = np.zeros_like(E_diffs)
local_weights[np.isclose(hamming_dists,1)] = 1
# normalise
row_sums = local_weights.sum(axis=1)
local_weights = local_weights / row_sums[:, np.newaxis]
print("local_weights:", local_weights)

uniform_weights = np.ones_like(E_diffs)
row_sums = uniform_weights.sum(axis=1)
uniform_weights = uniform_weights / row_sums[:, np.newaxis]
print("uniform_weights:", uniform_weights)


# brute force method to get QeMCMC weights
quantum_MCMC = QeMCMC(model, gamma=gamma, time=time, temp=1)
qemcmc_weights = np.zeros_like(E_diffs)
self_proposals = 0
for s in tqdm.tqdm(S):
    i = int(s,2)
    for j in range(reps):
        state = quantum_MCMC.get_s_prime(s)
        state_int = int(state,2)
        qemcmc_weights[i,state_int] += 1

"""# Hacky way to get QeMCMC weights
# CG doesnt work using this though
CM = CircuitMaker(model, gamma, time)
qemcmc_weights = np.zeros_like(E_diffs)
for i, s in enumerate(tqdm.tqdm(S)):
    for j in range(reps):
        state = CM.get_state_vector(s)
        #qemcmc_weights[int(i),:] += state.probabilities()
        sampled_strings = state.sample_counts(100)

        # this is very slow, lets speed it up.
        for bitstring, count in sampled_strings.items():
            state_int = int(bitstring,2)
            qemcmc_weights[int(i),state_int] += count
    qemcmc_weights[int(i),:] /= reps"""

# normalise
row_sums = qemcmc_weights.sum(axis=1)
qemcmc_weights = qemcmc_weights / row_sums[:, np.newaxis]
self_proposals_percent = (100*np.sum(np.diag(qemcmc_weights)))/(reps*(2**n_spins))
qemcmc_weights[np.diag_indices_from(qemcmc_weights)] = 0
print("self_proposals_percent:", self_proposals_percent)
print("qemcmc_weights:", qemcmc_weights)
#plot histograms of hamming distances and energy differences for each method
weights_list = [local_weights, uniform_weights, qemcmc_weights]
labels = ["local", "uniform", "QeMCMC"]
for weights, label in zip(weights_list, labels):
    plt.hist(hamming_dists.flatten(), bins = n_spins+1, weights = weights.flatten(), density=False, histtype="step",
            cumulative=True, label=label+"_hamming_dist")
    # plt.hist(E_diffs.flatten(), bins = n_spins, weights = weights.flatten(), density=True, histtype="step",
    #         cumulative=True, label=label+"_Energy_difference")
plt.legend()
plt.title("Hamming distance cummulative")
plt.xlabel("Hamming distance")
plt.ylabel("probability")
plt.show()



# plot energy difference histograms with area under curve for each method
for weights, label in zip(weights_list, labels):
    plt.hist(E_diffs.flatten(), bins = 2**(n_spins-2), weights = weights.flatten(), density=False, histtype="step",
            cumulative=True, label=label+"_Energy_difference")
plt.legend()
plt.title("Energy difference cummulative")
plt.xlabel("Energy difference")
plt.ylabel("probability")
plt.show()

