def main():
    print("Hello from qemcmc!")



from qemcmc import *



import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
print(plt.get_backend())

if __name__ == "__main__":
    
    
    
    def plot_chains(chains: list[MCMCChain], color: str, label:str):
        for chain in chains:
            energies = chain.get_current_energy_array()
            pos = chain.get_pos_array()
            plt.plot(pos, energies, color=color, alpha = 0.3)
        avg_energy = sum(chain.get_current_energy_array() for chain in chains) / len(chains)
        plt.plot(pos, avg_energy, color=color, label=f"Average {label}")
    
    
    n = 7
    steps = 50
    reps = 10
    model_type = "Fully Connected Ising"
    name = "Test Ising model"
    
    model = Model_Maker(n, model_type, name).model
    
    temp = 0.1
    initial_states = model.initial_state
    
    uni_chains = []
    for rep in range(reps):
        classical_uniform_MCMC = MCMC(model, temp, method = "uniform")
        uni_chain = classical_uniform_MCMC.run(steps, initial_state = initial_states[rep], name = "classical uniform MCMC", verbose = True, sample_frequency = 1)
        uni_chains.append(uni_chain)
    plot_chains(uni_chains, "orange", "classical uniform MCMC")
    
    loc_chains = []
    for rep in range(reps):
        classical_local_MCMC = MCMC(model, temp, method = "local")
        loc_chain = classical_local_MCMC.run(steps, initial_state = initial_states[rep], name = "classical local MCMC", verbose = True, sample_frequency = 1)
        loc_chains.append(loc_chain)
    plot_chains(loc_chains, "lightgreen", "classical local MCMC")
    
    def run_qemcmc(rep):
        qemcmc = QeMCMC(model, gamma=(0.3, 0.6), time=(2, 20), temp=temp)
        return qemcmc.run(steps, initial_state=initial_states[rep], name="QeMCMC", verbose=True, sample_frequency=1)

    Qe_chains = Parallel(n_jobs=-1)(delayed(run_qemcmc)(rep) for rep in range(reps))
    plot_chains(Qe_chains, "lightblue", "QeMCMC")
    
    plt.xlabel("MCMC step")
    plt.ylabel("Energy")
    plt.title("MCMC chains")
    plt.legend()
    plt.show()
    #plt.savefig("plots/MCMC_chains.png")
    
    

