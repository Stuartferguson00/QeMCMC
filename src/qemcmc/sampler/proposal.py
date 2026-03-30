# Internal
from abc import abstractmethod

from qemcmc.utils import MCMCChain, MCMCState, get_random_state
from qemcmc.model import EnergyModel

# External
import numpy as np
from typing import Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Proposal:
    """
    Base class for producing proposals for Markov Chain Monte Carlo algorithms.


    Subclasses implement the proposal mechanism by defining an
    ``update(state)`` method that generates a candidate state from the current one
    (e.g. single-spin flips, block updates, or quantum proposals).

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target distribution over spin configurations.
    temp : float
        Sampling temperature. The inverse temperature is stored as ``beta = 1 / temp``.

    Notes
    -----
    
    """

    def __init__(self, model: EnergyModel):
        """
        Initialize the MCMC routine for the Ising model.

        Args:
        model (EnergyModel): The energy function of the Ising model.
        """

        self.model = model
        self.n_spins = model.n_spins
        pass
    
    @abstractmethod
    def update(self, state: MCMCState) -> MCMCState:
        """
        Generate a candidate state from the current state using the proposal mechanism.
        This method should be implemented by subclasses to define the specific proposal strategy
        (e.g., single-spin flips, block updates, or quantum proposals).
        """
        pass