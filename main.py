def main():
    print("Hello from qemcmc!")


from QeMCMC.QeMCMC import QeMCMC
from QeMCMC.Model_Maker import Model_Maker
from QeMCMC.MCMC import MCMC
from matplotlib import pyplot as plt
from QeMCMC.helpers import MCMCChain
print(plt.get_backend())

if __name__ == "__main__":
    
    
    
    def plot_chains(chains: list[MCMCChain], color: str, label:str):
        for chain in chains:
            energies = chain.get_current_energy_array()
            pos = chain.get_pos_array()
            plt.plot(pos, energies, color=color, alpha = 0.3)
        avg_energy = sum(chain.get_current_energy_array() for chain in chains) / len(chains)
        plt.plot(pos, avg_energy, color=color, label=f"Average {label}")
    
    
    n = 10
    steps = 1000
    reps = 10
    model_type = "Fully Connected Ising"
    name = "Test Ising model"
    
    model = Model_Maker(n, model_type, name).model
    
    temp = 0.1
    initial_state = "0000000000"
    
    uni_chains = []
    for rep in range(reps):
        classical_uniform_MCMC = MCMC(model, temp, method = "uniform")
        uni_chain = classical_uniform_MCMC.run(steps, initial_state = initial_state, name = "classical uniform MCMC", verbose = True, sample_frequency = 1)
        uni_chains.append(uni_chain)
    plot_chains(uni_chains, "orange", "classical uniform MCMC")
    
    loc_chains = []
    for rep in range(reps):
        classical_local_MCMC = MCMC(model, temp, method = "local")
        loc_chain = classical_local_MCMC.run(steps, initial_state = initial_state, name = "classical local MCMC", verbose = True, sample_frequency = 1)
        loc_chains.append(loc_chain)
    plot_chains(loc_chains, "lightgreen", "classical local MCMC")
    
    Qe_chains = []
    for rep in range(reps):
        qemcmc = QeMCMC(model, gamma = (0.3,0.6), time = (2,20), temp = temp)
        Qe_chain = qemcmc.run(steps, initial_state = initial_state, name = "QeMCMC", verbose = True, sample_frequency = 1)
        Qe_chains.append(Qe_chain)
    plot_chains(Qe_chains, "lightblue", "QeMCMC")
    
    plt.xlabel("MCMC step")
    plt.ylabel("Energy")
    plt.title("MCMC chains")
    plt.legend()
    plt.show()
    plt.savefig("plots/MCMC_chains.png")
    
    

