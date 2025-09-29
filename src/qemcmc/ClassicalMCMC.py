import numpy as np
from .helpers import  get_random_state
from .energy_models import IsingEnergyFunction
from .MCMC import MCMC



class ClassicalMCMC(MCMC):
    """
    A class to perform Markov Chain Monte Carlo (MCMC) simulations for the Ising model.
    """
    
    def __init__(self, model: IsingEnergyFunction , temp, method = "uniform"):
        """
        Initialize the MCMC routine for the Ising model.
        
        Args:
        model (IsingEnergyFunction): The energy function of the Ising model.
        temp (float): The temperature of the system.
        method (str, optional): The update method to use. Options are "uniform" or "local". Default is "uniform".
        """
        super().__init__(model, temp)

        self.method = method

        if self.method == "uniform":
            self.update = self.update_uniform
        elif self.method == "local":
            self.update = self.update_local
        else:
            print("method name is incorrect. Choose from: 'uniform' or 'local'")
        

    def update_uniform(self,current_state_bitstring:str) -> str:
        """
        Updates the current state bitstring by generating a new random state bitstring of the same length.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: A new random state bitstring of the same length as the input.
        """
        s_prime = get_random_state(len(current_state_bitstring))
        return s_prime
    
    
    def update_local(self,current_state_bitstring:str)-> str:
        """
        Update the local state by flipping a randomly chosen spin in the current state bitstring.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: The new state bitstring after flipping a randomly chosen spin.
        """

        # Randomly choose which spin to flip
        choice = np.random.randint(0,self.n_spins)

        # Flip the chosen spin
        c_s = list(current_state_bitstring)
        c_s[choice] = str(int(c_s[choice]) ^ 1)
        
        # Return the new state as a bitstring
        s_prime = ''.join(c_s)   
        return s_prime

