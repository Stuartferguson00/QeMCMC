# Internal
from qemcmc.utils import get_random_state
from qemcmc.model import EnergyModel
from qemcmc.sampler import MCMC

# External
import numpy as np


class ClassicalMCMC(MCMC):
    """
    Classical Markov Chain Monte Carlo sampler.

    This class implements purely classical proposal mechanisms for MCMC.
    New candidate states are generated either by sampling a completely
    random (uniform) configuration or by performing a local single-spin flip.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target Boltzmann distribution.
    temp : float
        Sampling temperature of the system.
    method : str, optional
        Proposal mechanism used to generate candidate states.

        - ``"uniform"`` : propose a completely random spin configuration.
        - ``"local"`` : flip a single randomly chosen spin.

        Default is ``"uniform"``.
    """

    def __init__(self, model: EnergyModel, temp, method="uniform"):
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
            print("method must be 'uniform' or 'local'")

    def update_uniform(self, current_state_bitstring: str) -> str:
        """
        Updates the current state bitstring by generating a new random state bitstring of the same length.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: A new random state bitstring of the same length as the input.
        """
        s_prime = get_random_state(len(current_state_bitstring))
        return s_prime

    def update_local(self, current_state_bitstring: str) -> str:
        """
        Update the local state by flipping a randomly chosen spin in the current state bitstring.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: The new state bitstring after flipping a randomly chosen spin.
        """

        # Randomly choose which spin to flip
        choice = np.random.randint(0, self.n_spins)

        # Flip the chosen spin
        c_s = list(current_state_bitstring)
        c_s[choice] = str(int(c_s[choice]) ^ 1)

        # Return the new state as a bitstring
        s_prime = "".join(c_s)
        return s_prime
