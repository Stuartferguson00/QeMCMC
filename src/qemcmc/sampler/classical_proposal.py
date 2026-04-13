from qemcmc.model import EnergyModel
from qemcmc.sampler import Proposal
from qemcmc.utils import get_random_state
import numpy as np


class ClassicalProposal(Proposal):
    """
    Classical Markov Chain Monte Carlo proposer.

    This class implements purely classical proposal mechanisms for MCMC.
    New candidate states are generated either by sampling a completely
    random (uniform) configuration, or by performing a local single-spin or
    two-spin flip.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target Boltzmann distribution.
    method : str, optional
        Proposal mechanism used to generate candidate states.

        - ``"uniform"`` : propose a completely random spin configuration.
        - ``"local"`` : flip a single randomly chosen spin.
        - ``"2-local"`` : flip two randomly chosen spins.

        Default is ``"uniform"``.
    """

    def __init__(self, model: EnergyModel, method: str = "uniform"):
        super().__init__(model)
        self.method = method

        if self.method == "uniform":
            self.update = self.update_uniform
        elif self.method == "local":
            self.update = self.update_local
        elif self.method == "2-local":
            self.update = self.update_2local
        else:
            raise ValueError(f"Method '{method}' is not supported. Choose from 'uniform', 'local', or '2-local'.")

    def update_uniform(self, current_state_bitstring: str) -> str:
        """
        Proposes a new state by generating a random bitstring.

        Parameters
        ----------
        current_state_bitstring : str
            The current state represented as a bitstring (not used, but required by the API).

        Returns
        -------
        str
            A new random state bitstring of the same length.
        """
        return get_random_state(len(current_state_bitstring))

    def update_local(self, current_state_bitstring: str) -> str:
        """
        Proposes a new state by flipping a single randomly chosen spin.

        Parameters
        ----------
        current_state_bitstring : str
            The current state represented as a bitstring.

        Returns
        -------
        str
            The new state bitstring after flipping one spin.
        """
        # Randomly choose which spin to flip
        choice = np.random.randint(0, self.n_spins)

        # Flip the chosen spin
        c_s = list(current_state_bitstring)
        c_s[choice] = str(int(c_s[choice]) ^ 1)

        # Return the new state as a bitstring
        return "".join(c_s)

    def update_2local(self, current_state_bitstring: str) -> str:
        """
        Proposes a new state by flipping two randomly chosen spins.

        Parameters
        ----------
        current_state_bitstring : str
            The current state represented as a bitstring.

        Returns
        -------
        str
            The new state bitstring after flipping two spins.
        """
        # Randomly choose which two spins to flip
        choices = np.random.choice(self.n_spins, size=2, replace=False)

        # Flip the chosen spins
        c_s = list(current_state_bitstring)
        for choice in choices:
            c_s[choice] = str(int(c_s[choice]) ^ 1)

        # Return the new state as a bitstring
        return "".join(c_s)
