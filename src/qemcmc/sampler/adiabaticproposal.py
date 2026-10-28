import warnings

import numpy as np

from qemcmc.circuits import AdiabaticCircuitMaker
from qemcmc.coarse_grain import CoarseGraining
from qemcmc.model import EnergyModel
from qemcmc.sampler.proposal import Proposal

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_DELTA_T = 0.8


class AdiabaticProposal(Proposal):
    """
    Quantum-enhanced Markov Chain Monte Carlo proposer with adiabatic dressing.

    This class implements the adiabatic dressing proposal mechanism for quantum-enhanced
    MCMC (Ferguson & Wallden / Hsieh et al., arXiv:2603.28076). Candidate states are generated
    by simulating the quantum time evolution of a time-dependent transverse-field Hamiltonian:

        ``H(t) = h(t) * gamma * H_mixer + H_problem``

    The schedule function ``h(t)`` ramps the transverse field up from 0 to 1 over ramp
    duration ``alpha``, holds it at 1 for plateau duration ``kappa``, and ramps it down
    from 1 to 0 over duration ``alpha`` (total duration ``2 * alpha + kappa``).
    For long evolution times, adiabaticity preserves the ground/eigenstates, whereas for intermediate
    times diabatic transitions induce proposals concentrated near the target energy levels.

    The quantum proposal circuit is constructed using :class:`AdiabaticCircuitMaker`
    and can optionally operate on coarse-grained subgroups of spins to
    improve scalability.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target Boltzmann distribution.
    gamma : float | tuple[float, float]
        Transverse field strength. If a tuple is provided, a value is
        sampled uniformly from the range [min, max] at each step.
    alpha : float | tuple[float, float]
        Duration of the linear adiabatic ramp. If a tuple is provided, a value
        is sampled uniformly from the range [min, max] at each step.
    kappa : float | tuple[float, float]
        Duration of the constant plateau. If a tuple is provided, a value
        is sampled uniformly from the range [min, max] at each step.
    delta_t : float | None, optional
        Trotter step size for the approximate time evolution. The number of Trotter
        steps is derived as ``r = max(1, round((2 * alpha + kappa) / delta_t))``.
        Defaults to 0.8 if None.
    coarse_graining : CoarseGraining | None, optional
        A coarse-graining scheme to define spin subgroups. If None, no coarse-graining
        is used. Defaults to None.
    coupling_weights : list[float | tuple[float, float]] | None, optional
        Weights for the coupling tensors. If a tuple is provided, a weight is
        sampled uniformly from the range. Defaults to ones.

    Notes
    -----
    The proposal step simulates time evolution under a time-dependent transverse-field
    Hamiltonian and measures the resulting state to produce a candidate configuration.
    This proposal is then accepted or rejected using the Metropolis criterion
    to ensure convergence to the target Boltzmann distribution.
    """

    def __init__(
        self,
        model: EnergyModel,
        gamma: float | tuple[float, float],
        alpha: float | tuple[float, float],
        kappa: float | tuple[float, float],
        delta_t: float | None = None,
        coarse_graining: CoarseGraining | None = None,
        coupling_weights: list[float | tuple[float, float]] | None = None,
    ):
        super().__init__(model)

        if coupling_weights is not None:
            if len(coupling_weights) != len(model.normalised_couplings):
                raise ValueError(
                    f"Length of coupling_weights ({len(coupling_weights)}) must match the number of couplings in the model ({len(model.normalised_couplings)}), including constraint terms."
                )
            self.coupling_weights = coupling_weights
        else:
            self.coupling_weights = list(np.ones(len(model.normalised_couplings)))

        self.gamma = self._validate_gamma(gamma)
        self.alpha = self._validate_alpha(alpha)
        self.delta_t = self._validate_delta_t(delta_t)
        self.kappa = self._validate_kappa(kappa)

        self.method = "adiabatic_quantum"

        self.CM = AdiabaticCircuitMaker(self.model)

        self.cg = coarse_graining or CoarseGraining(model.n_spins)

    def update(self, current_state: str) -> str:
        """
        Perform sequential quantum updates across non-overlapping subgroups.

        Parameters
        ----------
        current_state : str
            Current state of the system as a bitstring.

        Returns
        -------
        str
            Updated state of the system as a bitstring.
        """
        if not isinstance(current_state, str):
            raise TypeError(f"Bitstring must be of type str, got {type(current_state)}: {current_state!r}")

        # Generate disjoint partitions of the spins
        partitions = self.cg.get_partitions()

        final_coupling_weights, g, alpha, kappa = self.sample_hyperparams()

        working_state = current_state
        for subgroup in partitions:
            # Recalculate local couplings as spins outside the subgroup may have flipped
            local_couplings = self.model.get_subgroup_couplings(
                subgroup=subgroup,
                current_state=working_state,
                coupling_weights=final_coupling_weights,
            )
            working_state = self.CM.update(
                s=working_state,
                subgroup_choice=subgroup,
                local_couplings=local_couplings,
                gamma=g,
                alpha=alpha,
                kappa=kappa,
                dt=self.delta_t if self.delta_t is not None else DEFAULT_DELTA_T,
            )
        return working_state

    def sample_hyperparams(self) -> tuple[list[float], float, float, float]:
        """
        Sample hyperparameters (coupling weights, gamma, alpha, kappa) for the current proposal step.

        Returns
        -------
        tuple[list[float], float, float, float]
            (final_coupling_weights, gamma, alpha, kappa) for this proposal step.
        """
        # Sample gamma, alpha, and kappa once per full proposal step
        g = self.gamma if not isinstance(self.gamma, tuple) else np.random.uniform(*self.gamma)
        alpha = self.alpha if not isinstance(self.alpha, tuple) else np.random.uniform(*self.alpha)
        kappa = self.kappa if not isinstance(self.kappa, tuple) else np.random.uniform(*self.kappa)

        # Pre-sample weights for all unique coupling weight settings (tuples or floats)
        unique_items = sorted(set(self.coupling_weights), key=str)
        unique_index = [[i for i, x in enumerate(self.coupling_weights) if x == item] for item in unique_items]

        sampled_weights = np.ones(len(unique_items), dtype=float)
        for i, item in enumerate(unique_items):
            if isinstance(item, (int, float)):
                sampled_weights[i] = item
            elif isinstance(item, tuple):
                sampled_weights[i] = np.random.uniform(*item)

        # Adjust the sampled weights by (1 - gamma) to balance their influence with the mixer term in the Hamiltonian
        #sampled_weights_adjusted_by_gamma = (1 - g) * sampled_weights  # Adjust weights by (1 - gamma) to balance with the mixer term
        # we no longer do this, as we are in the other convention

        # Construct the final list of coupling weights for this update step
        final_coupling_weights = np.ones(len(self.coupling_weights))
        for i, indices in enumerate(unique_index):
            for index in indices:
                final_coupling_weights[index] = sampled_weights[i]

        return final_coupling_weights, g, alpha, kappa

    def _validate_gamma(self, gamma: float | tuple[float, float]) -> float | tuple[float, float]:
        """
        Validate the weight parameter gamma.
        """
        if isinstance(gamma, (float, int)):
            if not 0.0 <= gamma:
                raise ValueError(f"gamma must be non-negative, got {gamma}")
            return float(gamma)

        if isinstance(gamma, (tuple, list)):
            if len(gamma) != 2:
                raise ValueError(f"gamma tuple or list must be (min, max), got {gamma}")
            g_min, g_max = gamma
            if not (0.0 <= g_min <= g_max):
                raise ValueError(f"gamma range must satisfy 0 <= min <= max, got {gamma}")
            return (float(g_min), float(g_max))

        raise TypeError(f"gamma must be a float or tuple[float, float] or list[float, float], got {type(gamma)}")

    def _validate_alpha(self, alpha: float | tuple[float, float]) -> float | tuple[float, float]:
        """
        Validate the adiabatic ramp duration alpha.
        """
        if isinstance(alpha, (int, float)):
            if alpha < 0:
                raise ValueError(f"alpha must be >= 0, got {alpha}")
            return float(alpha)

        if isinstance(alpha, (tuple, list)):
            if len(alpha) != 2:
                raise ValueError(f"alpha tuple must be (min, max), got {alpha}")
            t_min, t_max = alpha
            if t_min < 0 or t_max < 0:
                raise ValueError(f"alpha values must be >= 0, got {alpha}")
            if t_min > t_max:
                raise ValueError(f"alpha range must satisfy min <= max, got {alpha}")
            return (float(t_min), float(t_max))

        raise TypeError(f"alpha must be a float or tuple[float, float], got {type(alpha)}")

    def _validate_kappa(self, kappa: float | tuple[float, float]) -> float | tuple[float, float]:
        """
        Validate the plateau duration kappa.
        """
        if isinstance(kappa, (int, float)):
            if kappa < 0:
                raise ValueError(f"kappa must be >= 0, got {kappa}")
            return float(kappa)

        if isinstance(kappa, (tuple, list)):
            if len(kappa) != 2:
                raise ValueError(f"kappa tuple must be (min, max), got {kappa}")
            t_min, t_max = kappa
            if t_min < 0 or t_max < 0:
                raise ValueError(f"kappa values must be >= 0, got {kappa}")
            if t_min > t_max:
                raise ValueError(f"kappa range must satisfy min <= max, got {kappa}")
            return (float(t_min), float(t_max))

        raise TypeError(f"kappa must be a float or tuple[float, float], got {type(kappa)}")

    def _validate_delta_t(self, delta_t: float | None) -> float | None:
        """
        Validate the Trotter step size 'delta_t'.
        """
        if delta_t is None:
            return None
        if not isinstance(delta_t, (int, float)):
            raise TypeError(f"delta_t must be a float or None, got {type(delta_t)}")
        if delta_t <= 0:
            raise ValueError(f"delta_t must be positive, got {delta_t}")
        return float(delta_t)
