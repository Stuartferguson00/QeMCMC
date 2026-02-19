# Internal package imports
from qemcmc.sampler import MCMC
from qemcmc.model import EnergyModel
from qemcmc.circuits import PennyLaneCircuitMaker
from qemcmc.coarse_grain import CoarseGraining

# External package imports
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class QeMCMC(MCMC):
    """
    Class to set up the Quantum-enhanced Markov Chain Monte Carlo.
    """

    def __init__(
        self,
        model: EnergyModel,
        gamma: float | tuple[float],
        time: int | tuple[int],
        temp: float,
        delta_time: float = 0.8,
        coarse_graining=None,
    ):
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float]): The gamma parameter.
            time (int|tuple[int]): The time parameter. The number of trotter steps to take. (can be sampled from range represented by tuple.)
            temp (float): The temperature parameter.
            delta_time (float, optional): The delta time parameter for length of trotter steps. Defaults to 0.8.

            TODO: implement subgroups here instead of inside energy models.
        """

        super().__init__(model, temp)

        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.update = self.get_s_prime
        self.method = "quantum"

        # 1. Use Qiskit CircuitMaker
        # self.CM = CircuitMaker(self.model, self.gamma, self.time)

        # 2. Use Pennylane CircuitMaker
        self.CM = PennyLaneCircuitMaker(self.model, self.gamma, self.time)
        self.cg = coarse_graining or CoarseGraining(model.n)  # defaults to full system

    def get_s_prime(self, current_state: str) -> str:
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.

        Returns:
        str: The next state s_prime.
        """
        g = self.gamma
        t = self.time
        if isinstance(self.gamma, tuple):
            g = np.random.uniform(min(self.gamma), max(self.gamma))
        if isinstance(self.time, tuple):
            t = np.random.randint(min(self.time), max(self.time) + 1)

        subgroup_choice = self.cg.sample()
        local_couplings = self.model.get_subgroup_couplings(subgroup=subgroup_choice, current_state=current_state)

        self.CM.gamma = g
        self.CM.time = t
        self.CM.local_couplings = local_couplings

        # 1. Get s_prime using generic CircuitMaker
        # s_prime = self.CM.get_state(current_state)

        # 2. Get s_prime for coarse graining
        s_prime = self.CM.update(s=current_state, subgroup_choice=subgroup_choice)

        return s_prime
    
    
    def get_s_prime_alt(self, current_state: str) -> str:
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.

        Returns:
        str: The next state s_prime.
        """
        g = self.gamma
        t = self.time
        if isinstance(self.gamma, tuple):
            g = np.random.uniform(min(self.gamma), max(self.gamma))
        if isinstance(self.time, tuple):
            t = np.random.randint(min(self.time), max(self.time) + 1)

        subgroup_choice = self.cg.sample()
        local_couplings = self.model.get_subgroup_couplings(subgroup=subgroup_choice, current_state=current_state)

        self.CM.gamma = g
        self.CM.time = t
        self.CM.local_couplings = local_couplings

        # 1. Get s_prime using generic CircuitMaker
        # s_prime = self.CM.get_state(current_state)

        # 2. Get s_prime for coarse graining
        s_prime = self.CM.update_alt(s=current_state, subgroup_choice=subgroup_choice)

        return s_prime
