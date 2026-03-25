# Internal
from qemcmc.utils import MCMCChain, MCMCState, get_random_state
from qemcmc.model import EnergyModel

# External
import numpy as np
from typing import Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class MCMC:
    """
    Base class for Markov Chain Monte Carlo samplers.

    This class implements the common machinery for Metropolis-style sampling over
    discrete spin configurations defined by an :class:`EnergyModel`. It manages the
    MCMC loop, evaluates energies, performs Metropolis acceptance tests, and records
    the resulting Markov chain.

    Concrete subclasses implement the proposal mechanism by defining an
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
    The sampler targets the Boltzmann distribution

        p(s) ∝ exp(-E(s) / T)

    where ``E(s)`` is the energy of configuration ``s`` provided by the energy model.
    """

    def __init__(self, model: EnergyModel, temp: float):
        """
        Initialize the MCMC routine for the Ising model.

        Args:
        model (EnergyModel): The energy function of the Ising model.
        temp (float): The temperature of the system.
        """

        self.model = model
        self.temp = temp
        self.beta = 1 / self.temp
        self.n_spins = model.n_spins

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

        acceptance = min(1, exp_factor)
        return acceptance

    def test_accept(self, energy_s: float, energy_sprime: float, temperature: float = 1.0) -> MCMCState:
        """
        Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
        and s_init with probability 1-A.
        """
        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        # with warnings.catch_warnings():
        #    warnings.simplefilter("error", RuntimeWarning)
        try:
            exp_factor = np.exp(-delta_energy / temperature)
        except RuntimeWarning:
            if energy_sprime < energy_s:
                exp_factor = 1
            else:
                exp_factor = 0

            # print("Error in exponantial: delta_energy = ", delta_energy, "temperature = ", temperature, " energy_s = ", energy_s, " energy_sprime = ", energy_sprime)

        acceptance = min(1, exp_factor)  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

        return acceptance > np.random.rand()
