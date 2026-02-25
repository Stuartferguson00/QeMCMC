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
        gamma: float | tuple[float, float],
        time: int | tuple[int, int],
        temp: float,
        delta_time: float = 0.8,
        coarse_graining=None,
    ):
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float, float]): The gamma parameter.
            time (int|tuple[int, int]): The time parameter. The number of trotter steps to take. (can be sampled from range represented by tuple.)
            temp (float): The temperature parameter.
            delta_time (float, optional): The delta time parameter for length of trotter steps. Defaults to 0.8.
        """

        super().__init__(model, temp)

        self.gamma = self._validate_gamma(gamma)
        self.time = self._validate_time(time)
        self.delta_time = delta_time

        # what is this for?
        self.update = self.get_s_prime
        self.method = "quantum"

        self.CM = PennyLaneCircuitMaker(self.model, self.gamma, self.time, delta_time=self.delta_time)
        self.cg = coarse_graining or CoarseGraining(model.n)

    def get_s_prime(self, current_state: str) -> str:
        """
        Returns the next state s_prime based on the current state in the markov chain.

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

        s_prime = self.CM.update(s=current_state, subgroup_choice=subgroup_choice, local_couplings=local_couplings, gamma=g, time=t)

        return s_prime

    def _validate_gamma(self, gamma):
        if isinstance(gamma, (float, int)):
            if not (0.0 <= gamma <= 1.0):
                raise ValueError(f"gamma must be in [0, 1], got {gamma}")
            return float(gamma)

        if isinstance(gamma, tuple):
            if len(gamma) != 2:
                raise ValueError(f"gamma tuple must be (min, max), got {gamma}")
            g_min, g_max = gamma
            if not (0.0 <= g_min <= g_max <= 1.0):
                raise ValueError(f"gamma range must satisfy 0 ≤ min ≤ max ≤ 1, got {gamma}")
            return (float(g_min), float(g_max))

        raise TypeError(f"gamma must be a float or tuple[float, float], got {type(gamma)}")

    def _validate_time(self, time):
        if isinstance(time, int):
            if time <= 0:
                raise ValueError(f"time must be a positive integer, got {time}")
            return time

        if isinstance(time, tuple):
            if len(time) != 2:
                raise ValueError(f"time tuple must be (min, max), got {time}")
            t_min, t_max = time
            if not (isinstance(t_min, int) and isinstance(t_max, int)):
                raise TypeError(f"time range must contain integers, got {time}")
            if t_min <= 0 or t_max <= 0:
                raise ValueError(f"time values must be positive, got {time}")
            if t_min > t_max:
                raise ValueError(f"time range must satisfy min ≤ max, got {time}")
            return (t_min, t_max)

        raise TypeError(f"time must be an int or tuple[int, int], got {type(time)}")
