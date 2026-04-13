from abc import abstractmethod, ABC
from qemcmc.model import EnergyModel
from qemcmc.utils import MCMCState


class Proposal(ABC):
    """
    Abstract base class for producing proposals for Markov Chain Monte Carlo algorithms.

    Subclasses implement the proposal mechanism by defining an
    ``update(state)`` method that generates a candidate state from the current one
    (e.g. single-spin flips, block updates, or quantum proposals).

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target distribution over spin configurations.
    """

    def __init__(self, model: EnergyModel):
        self.model = model
        self.n_spins = model.n_spins

    @abstractmethod
    def update(self, state: MCMCState) -> MCMCState:
        """
        Generate a candidate state from the current state using the proposal mechanism.

        This method should be implemented by subclasses to define the specific proposal strategy
        (e.g., single-spin flips, block updates, or quantum proposals).

        Parameters
        ----------
        state : MCMCState
            The current state of the Markov chain.

        Returns
        -------
        MCMCState
            A new candidate state.
        """
        pass
