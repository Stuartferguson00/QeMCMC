# Internal
from qemcmc.sampler import Proposal
from qemcmc.model import EnergyModel
from qemcmc.circuits import CircuitMaker
from qemcmc.coarse_grain import CoarseGraining

# External
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class QeProposal(Proposal):
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
        delta_time: float = 0.8,
        coarse_graining=None,
        coupling_weights = None,
        m: int = 1,  # num of subgroups to partition into
    ):
        """
        Initializes an instance of the QeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float|tuple[float, float]): The gamma parameter.
            time (int|tuple[int, int]): The time parameter. The number of trotter steps to take. (can be sampled from range represented by tuple.)
            delta_time (float, optional): The delta time parameter for length of trotter steps. Defaults to 0.8.
            coarse_graining (CoarseGraining, optional): An optional coarse-graining object to define spin subgroups. Defaults to None.
            coupling_weights (list of float or list of tuple[float, float], optional): Optional list of weights for the coupling tensors in the model. 
                Length should match the number of coupling tensors in the model. Defaults to None (no weighting). If tuple provided, weight is sampled uniformly from the range for each coupling tensor at each step.
                Any identical coupling weights will be sampled together, so if you want to e.g. weight all the constraint couplings the same, you can just set those weights to be the same value or tuple, and this will be respected in the sampling.
            m (int, optional): The number of subgroups to partition the spins into for sequential updates. Defaults to 3.
        """

        super().__init__(model)

        if coupling_weights is not None:
            if len(coupling_weights) != len(model.normalised_couplings):
                raise ValueError(f"Length of coupling_weights must match number of couplings in the model. Expected {len(model.normalised_couplings)}, got {len(coupling_weights)}. Note that this includes constraint terms")
            self.coupling_weights = coupling_weights
        else:
            self.coupling_weights = list(np.ones(len(model.normalised_couplings)))

        self.gamma = self._validate_gamma(gamma)
        self.time = self._validate_time(time)
        self.delta_time = delta_time
        self.m = m

        #self.update = self.get_s_prime
        self.method = "quantum"

        self.CM = CircuitMaker(self.model, delta_time=self.delta_time)
        self.cg = coarse_graining or CoarseGraining(model.n)

    def update(self, current_state: str) -> str:
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

        unique_items = list(set(self.coupling_weights))
        # cant use np.where directly on list of floats and tuples, so have to do this weird list comprehension to get the indices of the unique items

        unique_index = [[i for i, x in enumerate(self.coupling_weights) if x == item] for item in unique_items]


        unique_items_floats = np.ones(len(self.coupling_weights), dtype=float) 
        for o, cw in enumerate(unique_items):           
            if isinstance(cw, (int, float)):
                unique_items_floats[o] = cw
            elif isinstance(cw, tuple):
                unique_items_floats[o] = np.random.uniform(*cw)


        coupling_weights = np.ones(len(self.coupling_weights))
        for o, unique_item in enumerate(unique_items):
            indices = unique_index[o]
            for index in indices:   
                coupling_weights[index] = unique_items_floats[o]



        working_state = current_state
        for subgroup in partitions:
            # recalculate couplings cause the spins outside the subgroup
            # might have flipped in the previous loop iteration
            local_couplings = self.model.get_subgroup_couplings(subgroup=subgroup, current_state=working_state, coupling_weights=coupling_weights)
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
