# Internal package imports
from qemcmc.sampler import MCMC
from qemcmc.model import EnergyModel
from qemcmc.circuits import PennyLaneCircuitMaker

# External package imports
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

            TODO: implement subgroups here instead of inside energy models.
        """

        super().__init__(model, temp)

        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.update = self.get_s_prime
        self.method = "quantum"

        # Initialize quantum circuit here
        # Uncomment either 1. or 2. depending on which CM you want to use.

        # 1. Use Qiskit CircuitMaker
        # self.CM = CircuitMaker(self.model, self.gamma, self.time)

        # 2. Use Pennylane CircuitMaker
        self.CM = PennyLaneCircuitMaker(self.model, self.gamma, self.time)

    def get_s_prime(self, current_state: str) -> str:
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.

        Returns:
        str: The next state s_prime.
        """

        # Uncomment either 1. or 2. depending on which CM you want to use and if you want to do CG.

        # 1. Get s_prime using generic CircuitMaker
        # s_prime = self.CM.get_state(current_state)

        # 2. Get s_prime for coarse graining
        s_prime = self.CM.update(current_state)

        return s_prime
