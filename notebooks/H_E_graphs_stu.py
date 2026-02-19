import itertools
import matplotlib.pyplot as plt
from qemcmc.model import EnergyModel
from qemcmc.circuits import PennyLaneCircuitMaker as CircuitMaker
from qemcmc.utils import ModelMaker
from qemcmc.sampler import QeMCMC
import numpy as np
import tqdm
import joblib
from qemcmc.coarse_grain import CoarseGraining

# Define parameters
n_spins = 8  # Number of spins in the system
reps = 10
coarse_graining_number = n_spins
# QeMCMC parameters
gamma = 0.45#(0.3,0.6)#0.4  # (0.3,0.6)# Relative strength of mixer hamiltonian
time = 10#(1,20)#10  # (1,20) # Time for hamiltonian simulation


# Make all combinations of subgroups that include coarse_graining_number of integers between 0 and n_spins -1
# Feroz, could we add this simple functionality to ModelMaker?
subgroups = list(itertools.combinations(range(n_spins), coarse_graining_number))


shape_of_J = (n_spins, n_spins)
J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
J_tril = np.tril(J, -1)
J_triu = J_tril.transpose()
J = J_tril# + J_triu

h = np.round(np.random.normal(0, 1, n_spins), decimals=4)

couplings = [h, J]
# why does the user have to calculate their own alpha? At least we should do it in model maker. Ask the user to input max_number of qubits, and we enumerate all combinations internally.
alpha = np.sqrt(n_spins) / np.sqrt(sum([J[i][j] ** 2 for i in range(n_spins) for j in range(i)]) + sum([h[j] ** 2 for j in range(n_spins)]))

model = EnergyModel(n=n_spins, couplings=couplings, alpha=alpha)
CG = CoarseGraining(n = n_spins, subgroups= subgroups)
"""
model_type = "Coarse Grained Ising"
name = "Test Ising model"
model = ModelMaker(n_spins, model_type, name).model

"""


ell_enegries = model.get_all_energies()

S = np.array([np.binary_repr(s, width=n_spins) for s in range(2**n_spins)])
S_integer_arr = np.array([[int(bit) for bit in state] for state in S])
E_s = np.array(ell_enegries)

# find matrices of Eneegry differences of the configyurations, and hamming distances
# not efficient
E_diffs = np.zeros((2**n_spins, 2**n_spins))
hamming_dists = np.zeros((2**n_spins, 2**n_spins))
for s in S:
    for s_prime in S:
        hamming_dist = np.sum(np.bitwise_xor(S_integer_arr[int(s, 2)], S_integer_arr[int(s_prime, 2)]))
        E_diff = abs(E_s[int(s, 2)] - E_s[int(s_prime, 2)])
        E_diffs[int(s, 2), int(s_prime, 2)] = E_diff
        hamming_dists[int(s, 2), int(s_prime, 2)] = hamming_dist



# Weights are the probability of proposal from s to s'
local_weights = np.zeros_like(E_diffs)
local_weights[np.isclose(hamming_dists, 1)] = 1
# normalise
row_sums = local_weights.sum(axis=1)
local_weights = local_weights / row_sums[:, np.newaxis]

uniform_weights = np.ones_like(E_diffs)
row_sums = uniform_weights.sum(axis=1)
uniform_weights = uniform_weights / row_sums[:, np.newaxis]

def bit_reverse_bitwise(value, n):
    reversed_val = 0
    for i in range(n):
        # Extract the bit at position i and shift it to position (n - 1 - i)
        if (value >> i) & 1:
            reversed_val |= (1 << (n - 1 - i))
    return reversed_val

# check individual proposals from QeMCMC

expectedstate = S[11]
qemcmcm_test = QeMCMC(model, gamma=0, time=1, temp=1, coarse_graining=CG)
for i in range(10):
    state = qemcmcm_test.get_s_prime(expectedstate)
    state_int = int(state, 2)
    print("state_int:", state_int)
print("expected state:", expectedstate)
print("expected state int:", int(expectedstate, 2), n_spins)


"""statevector = qemcmcm_test.CM.get_state(expectedstate)
print("statevector:", statevector)
qemcmcm_test.get_s_prime(expectedstate)"""

# brute force method to get QeMCMC weights
quantum_MCMC = QeMCMC(model, gamma=gamma, time=time, temp=1)
qemcmc_weights_brute = np.zeros_like(E_diffs)
self_proposals = 0
for i, s in enumerate(tqdm.tqdm(S)):
    for j in range(reps):
        state = quantum_MCMC.get_s_prime_alt(s)
        #state_int = bit_reverse_bitwise(int(state, 2), n_spins)
        state_int = int(state, 2)
        qemcmc_weights_brute[i, state_int] += 1
qemcmc_weights_brute = qemcmc_weights_brute / (reps)

# Hacky way to get QeMCMC weights
# CG doesnt work using this though
def inner_loop(s):
    if type(gamma) is not float:
        gamma_ = np.random.uniform(gamma[0], gamma[1])
    else:
        gamma_ = gamma
    if type(time) is not int:
        time_ = np.random.randint(time[0], time[1] + 1)
    else:
        time_ = time
    
    state = quantum_MCMC.CM.get_state_vector(s)
    #qemcmc_weights[int(i),:] += state.probabilities()
    #sampled_strings = state.sample_counts(100)
    # for bitstring, count in sampled_strings.items():
    #     state_int = int(bitstring,2)
    #     qemcmc_weights[int(i),state_int] += count
    return np.abs(state) **2

qemcmc_weights = np.zeros_like(E_diffs)
for i, s in enumerate(tqdm.tqdm(S)):
    result = inner_loop(s)
    qemcmc_weights[int(i),:] += result


# plot the 2^n x 2^n grid, as a heat map, where the x axis is s, y axis is s', and the color is the weight of proposal from s to s'
# but first, reorder the weights so that the x and y axis are in order of increasing E(s)
E_order = np.argsort(E_s)
qemcmc_weights_ordered = qemcmc_weights[E_order, :][:, E_order]
qemcmc_weights_brute_ordered = qemcmc_weights_brute[E_order, :][:, E_order]
# make the color graident logarithmic
plt.hist2d(np.repeat(np.arange(2**n_spins), 2**n_spins), np.tile(np.arange(2**n_spins), 2**n_spins), bins=50, weights=qemcmc_weights_ordered.flatten(), norm="log")
plt.title("QeMCMC proposal weights heatmap")
plt.xlabel("s")
plt.ylabel("s'")
plt.colorbar(label="Proposal weight")
plt.show()


plt.hist2d(np.repeat(np.arange(2**n_spins), 2**n_spins), np.tile(np.arange(2**n_spins), 2**n_spins), bins=50, weights=qemcmc_weights_brute_ordered.flatten(), norm="log")
plt.title("QeMCMC proposal weights heatmap")
plt.xlabel("s")
plt.ylabel("s'")
plt.colorbar(label="Proposal weight")
plt.show()
















self_proposals_percent = (100 * np.sum(np.diag(qemcmc_weights))) / (reps * (2**n_spins))

# normalise
row_sums = qemcmc_weights.sum(axis=1)
qemcmc_weights = qemcmc_weights / row_sums[:, np.newaxis]
#qemcmc_weights[np.diag_indices_from(qemcmc_weights)] = 0
print("self_proposals_percent:", self_proposals_percent)
print("qemcmc_weights:", qemcmc_weights)

row_sums_brute = qemcmc_weights_brute.sum(axis=1)
qemcmc_weights_brute = qemcmc_weights_brute / row_sums_brute[:, np.newaxis]

# plot histograms of hamming distances and energy differences for each method
weights_list = [local_weights, uniform_weights, qemcmc_weights, qemcmc_weights_brute]
labels = ["local", "uniform", "QeMCMC", "QeMCMC_brute"]
for weights, label in zip(weights_list, labels):
    plt.hist(hamming_dists.flatten(), bins=2**n_spins, weights=weights.flatten(), density=False, histtype="step", cumulative=True, label=label + "_hamming_dist")
    # plt.hist(E_diffs.flatten(), bins = n_spins, weights = weights.flatten(), density=True, histtype="step",
    #         cumulative=True, label=label+"_Energy_difference")
plt.legend()
plt.title("Hamming distance cummulative")
plt.xlabel("Hamming distance")
plt.ylabel("probability")
plt.show()


# plot energy difference histograms with area under curve for each method
for weights, label in zip(weights_list, labels):
    plt.hist(E_diffs.flatten(), bins=2 ** (n_spins), weights=weights.flatten(), density=False, histtype="step", cumulative=True, label=label + "_Energy_difference")
plt.legend()
plt.title("Energy difference cummulative")
plt.xlabel("Energy difference")
plt.ylabel("probability")
plt.show()
