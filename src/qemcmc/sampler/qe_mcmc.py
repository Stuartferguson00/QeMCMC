# Internal
from qemcmc.sampler import MCMC
from qemcmc.model import EnergyModel
from qemcmc.circuits import CircuitMaker
from qemcmc.coarse_grain import CoarseGraining

# External
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class QeMCMC(MCMC):
    """
    Quantum-enhanced Markov Chain Monte Carlo sampler.

    This class implements the proposal mechanism of the quantum-enhanced
    MCMC algorithm, where candidate states are generated via simulated
    quantum time evolution of a transverse-field Hamiltonian. The resulting
    proposals are then accepted or rejected using the classical Metropolis
    rule defined in the base :class:`MCMC` class.

    The quantum proposal circuit is constructed using :class:`CircuitMaker`
    and can optionally operate on coarse-grained subgroups of spins to
    improve scalability.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target Boltzmann distribution.
    gamma : float or tuple[float, float]
        Transverse field strength used in the quantum evolution. If a tuple
        is provided, a value is sampled uniformly from the range at each step.
    time : int or tuple[int, int]
        Number of Trotter steps used in the quantum evolution. If a tuple is
        provided, the number of steps is randomly sampled within the range.
    temp : float
        Sampling temperature of the system.
    delta_time : float, optional
        Length of each Trotter step in the quantum evolution. Default is 0.8.
    coarse_graining : CoarseGraining, optional
        Optional coarse-graining scheme defining spin subgroups on which the
        quantum proposal acts.
    m: int, optional
        Number of subgroups to partition the spins into for sequential updates. Default is 3.

    Notes
    -----
    The proposal step simulates time evolution of a Quantum circuit of a
    Hamiltonian and measures the resulting state to produce a candidate
    configuration. This proposal is then accepted or rejected using the
    Metropolis criterion to ensure convergence to the target Boltzmann
    distribution.
    """

    def __init__(
        self,
        model: EnergyModel,
        gamma: float | tuple[float, float],
        time: int | tuple[int, int],
        temp: float,
        delta_time: float = 0.8,
        coarse_graining=None,
        m: int = 3,  # num of subgroups to partition into
    ):
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float, float]): The gamma parameter.
            time (int|tuple[int, int]): The time parameter. The number of trotter steps to take. (can be sampled from range represented by tuple.)
            temp (float): The temperature parameter.
            delta_time (float, optional): The delta time parameter for length of trotter steps. Defaults to 0.8.
            coarse_graining (CoarseGraining, optional): An optional coarse-graining object to define spin subgroups. Defaults to None.
            m (int, optional): The number of subgroups to partition the spins into for sequential updates. Defaults to 3.
        """

        super().__init__(model, temp)

        self.gamma = self._validate_gamma(gamma)
        self.time = self._validate_time(time)
        self.delta_time = delta_time
        self.m = m

        # check
        self.update = self.get_s_prime
        self.method = "quantum"

        self.CM = CircuitMaker(self.model, self.gamma, self.time, delta_time=self.delta_time)
        self.cg = coarse_graining or CoarseGraining(model.n)

    def get_s_prime(self, current_state: str) -> str:
        """
        Perform 'm' sequential quantum updates across
        non-overlapping subgroups to produce a macro-proposal s_prime.
        """
        if not isinstance(current_state, str):
            raise TypeError(f"Bitstring must be of type str, got {type(current_state)}: {current_state!r}")

        # I only sample gamma and time once per full proposal, not per subgroup update
        g = self.gamma if not isinstance(self.gamma, tuple) else np.random.uniform(*self.gamma)
        t = self.time if not isinstance(self.time, tuple) else np.random.randint(self.time[0], self.time[1] + 1)

        # Generate m disjoint partitions (e.g., if n=10, m=3 -> [[3,8,1,9], [0,4,2], [5,6,7]])
        partitions = self.cg.get_partitions(m=self.m)

        working_state = current_state
        for subgroup in partitions:
            # recalculate couplings cause the spins outside the subgroup
            # might have flipped in the previous loop iteration
            local_couplings = self.model.get_subgroup_couplings(subgroup=subgroup, current_state=working_state)
            working_state = self.CM.update(s=working_state, subgroup_choice=subgroup, local_couplings=local_couplings, gamma=g, time=t)

        return working_state

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
