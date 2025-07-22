##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from typing import Optional
from tqdm import tqdm
from .helpers import MCMCChain, MCMCState, get_random_state
from .energy_models import IsingEnergyFunction
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)






class MCMC:
    """
    A base class to perform Markov Chain Monte Carlo (MCMC) simulations for the Ising model.
    
    """
    
    
    def __init__(self, model: IsingEnergyFunction , temp: float):
        """
        Initialize the MCMC routine for the Ising model.
        
        Args:
        model (IsingEnergyFunction): The energy function of the Ising model.
        temp (float): The temperature of the system.
        
        
        """


        self.model = model
        self.temp = temp
        self.beta = 1/self.temp
        self.n_spins = model.num_spins
        
        





    def run(self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name:str = "MCMC",
        verbose:bool = False,
        sample_frequency:int = 1):



        """
        Run the classical MCMC algorithm for a specified number of hops.
        Parameters:
        n_hops : int
            The number of hops (iterations) to perform in the MCMC algorithm.
        initial_state : Optional[str], optional
            The initial state to start the MCMC algorithm. If None, a random state is generated. Default is None.
        name : str, optional
            The name of the MCMC run. Default is "classical MCMC".
        verbose : bool, optional
            If True, prints detailed information during the run. Default is False.
        sample_frequency : int, optional
            The frequency at which to sample and store states in the MCMC chain. Default is 1.
        Returns:
        MCMCChain
            The MCMC chain containing the sequence of states visited during the run.
        """








        if name is None:
            name = self.method + " MCMC"

        # Either get a random state or use initial state given
        if initial_state is None:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True, position = 0)
        else:
            initial_state = MCMCState(initial_state, accepted=True, position = 0)
        
        #set initial state
        current_state: MCMCState = initial_state
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state.energy = energy_s


        if verbose:
            print("starting with: ", current_state.bitstring, "with energy:", energy_s)


        #define MCMC chain
        mcmc_chain = MCMCChain([current_state], name= name)
        
        
        # Do MCMC 
        for i in tqdm(range(0, n_hops), desc='Run '+name, disable= not verbose ):

            # Propose a new state
            s_prime = self.update(current_state.bitstring)
            
            # Find energy of the new state
            energy_sprime = self.model.get_energy(s_prime)
            
            # Decide whether to accept the new state
            accepted = self.test_accept(energy_s, energy_sprime, temperature=self.temp)
            
            # If accepted, update current_state
            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position = i)
                
                
                
            # if time to sample, add state to chain
            if i//sample_frequency == i/sample_frequency and i != 0 :
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position = i))
                
            

        return mcmc_chain





    def test_probs(self, energy_s: float, energy_sprime: float) -> float:
        
        """
        Calculate the probability ratio between two states based on their energies.
        This function computes the exponential factor used in the Metropolis-Hastings 
        algorithm to determine the acceptance probability of a new state s' given 
        the current state s. The probability ratio is calculated as exp(-(E(s') - E(s)) / T),
        where E(s) and E(s') are the energies of the current and proposed states, respectively,
        and T is the temperature.
        Args:
            energy_s (float): The energy of the current state s.
            energy_sprime (float): The energy of the proposed state s'.
        Returns:
            float: The probability ratio exp(-(E(s') - E(s)) / T).
        """

        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        if energy_sprime < energy_s:
            exp_factor = 1
        else:
            exp_factor = np.exp(-delta_energy / self.temp)
            
        acceptance = min(
            1, exp_factor
        )  
        return acceptance

    
    def test_accept(self, 
        energy_s: float, energy_sprime: float, temperature: float = 1.
        ) -> MCMCState:
        """
        Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
        and s_init with probability 1-A.
        """
        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        #with warnings.catch_warnings():
        #    warnings.simplefilter("error", RuntimeWarning)
        try:
            exp_factor = np.exp(-delta_energy / temperature)
        except RuntimeWarning:
            if energy_sprime < energy_s:
                exp_factor = 1
            else:
                exp_factor = 0
            
            #print("Error in exponantial: delta_energy = ", delta_energy, "temperature = ", temperature, " energy_s = ", energy_s, " energy_sprime = ", energy_sprime)
                
        acceptance = min(
            1, exp_factor
        )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

        return acceptance > np.random.rand()
