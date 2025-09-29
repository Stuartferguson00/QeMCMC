import numpy as np
from .MCMC import MCMC
from .energy_models import IsingEnergyFunction, EnergyModel
from .CircuitMaker import CircuitMakerIsing
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class QeMCMC(MCMC):
    """
    Class to set up the Quantum-enhanced Markov Chain Monte Carlo.
    """
    def __init__(self, model:EnergyModel, gamma:float|tuple[float], time: int|tuple[int], temp: float, delta_time:float = 0.8):
        #havent done type hinting yet
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float]): The gamma parameter.
            time (int|tuple[int]): The time parameter. The number of trotter steps to take. (can be sampled from range represented by tuple.)
            temp (float): The temperature parameter.
            delta_time (float, optional): The delta time parameter for length of trotter steps. Defaults to 0.8.
        """

        super().__init__(model, temp)
        
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.update = self.get_s_prime
        self.method = "quantum"
        
    
    def get_s_prime(self, current_state: str)-> str:
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
        CM = CircuitMakerIsing(self.model, g, t)
        s_prime = CM.get_state_obtained_binary(current_state)
        
        return s_prime


    def get_output_statevector(self, current_state: str)-> str:
        """
        Returns the statevector of possible s_primes based on the current state, g, and t.

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
            raise TypeError("gamma must be a float in get_output_statevector")
            
        if type(self.time) is int:
            t = self.time
        elif type(self.time) is tuple:
            t = np.random.randint(low= np.min(self.time), high = np.max(self.time),size = 1)[0]
        else:
            raise TypeError("time must be an int in get_output_statevector")
        
        
        
        # Get s_prime
        CM = CircuitMakerIsing(self.model, g, t)
        s_primes = CM.get_statevector_obtained(current_state)
        
        return s_primes


    
