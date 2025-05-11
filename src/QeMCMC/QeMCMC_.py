##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from tqdm import tqdm
from .helpers import MCMCChain, MCMCState, get_random_state, test_accept
from .energy_models import IsingEnergyFunction
from typing import Optional, Union, Tuple
from .CircuitMaker import CircuitMaker




class QeMCMC:
    """
    Class to set up the Quantum-enhanced Markov Chain Monte Carlo.
    
    """
    
    def __init__(self, model:IsingEnergyFunction, gamma:float|tuple[float], time: int|tuple[int], temp: float, delta_time:float = 0.8):
        #havent done type hinting yet
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float]): The gamma parameter.
            time (int|tuple[int]): The time parameter.
            temp (float): The temperature parameter.
            delta_time (float, optional): The delta time parameter. Defaults to 0.8.
        """
        self.model = model
        self.n_spins = model.num_spins
        
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        
        self.temp = temp
        self.beta = 1 / self.temp
        

        
        



    def run(self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name:str = "QeMCMC",
        verbose:bool = False,
        sample_frequency:int = 1):
        """
        Runs the quantum MCMC algorithm for a specified number of hops.

        Args:
            n_hops (int): The number of hops to perform in the MCMC algorithm.
            initial_state (Optional[str], optional): The initial state for the MCMC algorithm. If not provided, a random state will be generated. Defaults to None.
            name (str, optional): The name of the MCMC chain. Defaults to "quMCMC".
            verbose (bool, optional): Whether to print verbose output during the algorithm execution. Defaults to False.
            sample_frequency (int, optional): The frequency at which to sample states. Defaults to 1.

        Returns:
            MCMCChain: The MCMC chain containing the states collected during the algorithm execution.
        """



        # Either get a random state or use initial state given
        if initial_state is None:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True, position = 0)
        else:
            initial_state = MCMCState(initial_state, accepted=True, position = 0)
        
        if len(initial_state.bitstring) != self.n_spins:
            raise ValueError(f"Initial state must be of length {self.n_spins}, but got {len(initial_state.bitstring)}")
        
        #set initial state
        current_state: MCMCState = initial_state
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state.energy = energy_s


        if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)



        
        # Define chain
        mcmc_chain = MCMCChain([current_state], name= name)

        # Do MCMC
        for i in tqdm(range(0, n_hops), desc='running QeMCMC', disable= not verbose ):
            
            # Propose a new state
            s_prime = self.get_s_prime(current_state.bitstring)

            #Find energy of the new state
            energy_sprime = self.model.get_energy(s_prime)
            # Decide whether to accept the new state
            accepted = test_accept(energy_s, energy_sprime, temperature=self.temp)
            


            # If accepted, update current_state
            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_sprime, position = i)
                energy_s = energy_sprime

            # if time to sample, add state to chain
            if i//sample_frequency == i/sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position = i))
            
            
        return mcmc_chain
    
    def get_s_prime(self, current_state: str):
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.

        Returns:
        str: The next state s_prime.
        """
        # Should probably type check time and gamma in init not here 
        if type(self.gamma) is float or type(self.gamma) is int:
            g = self.gamma
        elif type(self.gamma) is tuple:
            g = np.round(np.random.uniform(low= min(self.gamma), high = max(self.gamma),size = 1), decimals=6)[0]
        else:
            raise TypeError("gamma must be either a float or a tuple")
            
        if type(self.time) is int:
            t = self.time
        elif type(self.time) is tuple:
            t = np.random.randint(low= np.min(self.time), high = np.max(self.time),size = 1)[0]
        else:
            raise TypeError("time must be either an int or a tuple")
        
        
        
        # Get s_prime
        CM = CircuitMaker(self.model, g, t)
        s_prime = CM.get_state_obtained_binary(current_state)
        
        return s_prime


    
