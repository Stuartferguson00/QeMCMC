import numpy as np
from .MCMC import MCMC
from .energy_models import EnergyModel
from .CircuitMaker import CircuitMakerIsing, CircuitMaker
import warnings

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
    ):
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

        if type(self.gamma) is float or type(self.gamma) is int:
            g = self.gamma
        elif type(self.gamma) is tuple:
            g = np.round(
                np.random.uniform(low=min(self.gamma), high=max(self.gamma), size=1),
                decimals=6,
            )[0]
        else:
            raise TypeError("gamma must be either a float or a tuple")

        if type(self.time) is int:
            t = self.time
        elif type(self.time) is tuple:
            t = np.random.randint(low=np.min(self.time), high=np.max(self.time), size=1)[0]
        else:
            raise TypeError("time must be either an int or a tuple")

        # Initialize quantum circuit here instead of inside get_s_prime each time.
        # Uncomment either 1. or 2. depending on which CM you want to use.

        # 1. Use CircuitMakerIsing
        # self.CM = CircuitMakerIsing(self.model, g, t)

        # 2. Use generic CircuitMaker
        self.CM = CircuitMaker(self.model, g, t)

    def get_s_prime(self, current_state: str) -> str:
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.

        Returns:
        str: The next state s_prime.
        """

        # Uncomment either 1. 2. or 3. depending on which CM you want to use and if you want to do CG.

        # 1. Get s_prime using CircuitMakerIsing
        # s_prime = self.CM.get_state_obtained_binary(current_state)

        # 2. Get s_prime using generic CircuitMaker
        s_prime = self.CM.get_state(current_state)

        # 3. Get s_prime for coarse graining
        # s_prime = self.CM.update(current_state)

        return s_prime
