import typing
import numpy as np
import dimod
from qemcmc.utils.helpers import get_random_state
import warnings
from .energy_model import EnergyModel


class ConstraintModel(EnergyModel):
    """
    A subclass of EnergyModel that incorporates a constraint function to define valid configurations.

    The constraint function takes a state as input and returns True if the state is valid
    (satisfies the constraint) and False otherwise. The energy of invalid states is set to
    infinity, effectively excluding them from the Boltzmann distribution.

    Parameters
    ----------
    n : int
        Number of spins in the model.
    constraint_couplings : list[dimod.BinaryPolynomial]
        List of coupling tensors (dimod.BinaryPolynomial) defining the constraint.
    constraint_signs : list[float]
        Sign convention(s) for the constraint couplings.
    couplings : list[dimod.BinaryPolynomial]
        List of coupling tensors (dimod.BinaryPolynomial) defining the energy function.
    constraint_func : typing.Callable[[str], bool]
        A function that takes a state (string representation of spin configuration)
        and returns True if the state satisfies the constraint, and False otherwise.
    name : str, optional
        Optional label for the model (used in plotting / logging).
    cost_function_signs : list[float], optional
        Sign convention(s) used by downstream components (e.g. proposal/acceptance conventions).
    model_type : str, optional
        Type of model, either 'ising' or 'binary'. This determines how the binary states are interpreted
        and how the energy is calculated. 'ising' models use spin values {-1, +1}, while 'binary'
        models use binary values {0, 1}.

    Notes
    -----
    - The energy of any state that does not satisfy the constraint is set to infinity, which means
      such states will have zero probability in the Boltzmann distribution.
    - This class can be used to model systems with hard constraints on the configurations, such as
      certain combinatorial optimization problems or physical systems with forbidden states.
    """

    def __init__(self, n: int, constraint_couplings: list[dimod.BinaryPolynomial], constraint_signs: list[float], couplings: list[dimod.BinaryPolynomial], constraint_func: typing.Callable[[str], bool], **kwargs):
        if not callable(constraint_func):
            raise ValueError("constraint_func must be a callable function that takes a state as input and returns True/False.")
        self.constraint_func = constraint_func

        # This model requires a special way to generate initial states that respect the constraint.
        self.get_initial_states = self.get_initial_states_constraint

        if couplings is not None:
            super().__init__(n=n, couplings=couplings, **kwargs)
        else:
            super().__init__(n=n, couplings=[], **kwargs)

        self.constraint_couplings = constraint_couplings
        self.constraint_signs = constraint_signs

        # Calculate normalization factors for constraint couplings
        self.constraint_coupling_alpha = self.calculate_alpha(n, constraint_couplings)

        # These are the couplings used in quantum proposals
        # Combine and normalize the energy and constraint couplings
        self.normalised_couplings = (
            [dimod.BinaryPolynomial({k: v * self.alpha for k, v in poly.items()}, poly.vartype) for i, poly in enumerate(self.couplings)]
            +
            [dimod.BinaryPolynomial({k: v * self.constraint_coupling_alpha[i] for k, v in poly.items()}, poly.vartype) for i, poly in enumerate(self.constraint_couplings)]
        )

        # Store the un-normalized total couplings
        self.total_couplings = self.couplings + self.constraint_couplings

    def get_initial_states_constraint(self, num_initial_states: int):
        """
        Generates a list of random initial states that satisfy the constraint function.

        Parameters
        ----------
        num_initial_states:
            The number of initial states to generate.

        Returns
        -------
        list:
            A list of random initial states that are valid according to the constraint.
        """
        init_states = []
        counter = 0
        max_attempts = 1000
        while len(init_states) < num_initial_states and counter < max_attempts:
            state = get_random_state(self.n_spins)
            if self.constraint_func(state):
                init_states.append(state)
            counter += 1

        if len(init_states) < num_initial_states:
            warnings.warn(
                f"Could not find enough valid initial states satisfying the constraint. "
                f"Found {len(init_states)} valid states after {max_attempts} attempts. "
                f"You may want to provide them manually if more are needed."
            )
        return init_states

    def get_constraint_energy(self, state: str) -> float:
        """
        Calculate the energy contribution from the constraint couplings for a given state.

        Parameters
        ----------
        state : str
            The state for which to calculate the constraint energy.

        Returns
        -------
        float:
            The energy contribution from the constraint couplings for the given state.
        """
        return self.calculate_energy(state, self.constraint_couplings, self.constraint_signs)

    def get_total_energy(self, state: str) -> float:
        """
        Calculate the total energy of a given state, including both the regular energy and the constraint energy.

        Parameters
        ----------
        state : str
            The state for which to calculate the total energy.

        Returns
        -------
        float:
            The total energy of the given state, including contributions from both the regular couplings and the constraint couplings.
        """
        return self.get_energy(state) + self.get_constraint_energy(state)
