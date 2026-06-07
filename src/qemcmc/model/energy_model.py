import itertools
import typing
from typing import List
import numpy as np
import dimod
import math
from tqdm import tqdm
from collections import defaultdict
from qemcmc.utils.helpers import get_random_state


class EnergyModel:
    """
    A base class for energy models for a classical energy function over n spins,
    defined by dimod.BinaryPolynomial coupling tensors.

    Parameters
    ----------
    n:
        Number of spins
    couplings:
        List of dimod.BinaryPolynomial representing interaction tensors.
    name:
        Optional label for the model (used in plotting / logging).
    cost_function_signs:
        Sign convention(s) used by downstream components (e.g. proposal/acceptance conventions).
        For example, `[-1, -1]` for linear and quadratic terms.
    model_type: str
        Type of model, either 'ising' or 'binary'. This determines how the binary states are interpreted
        and how the energy is calculated. 'ising' models use spin values {-1, +1}, while 'binary'
        models use binary values {0, 1}.

    Notes
    -----
    - Energies are computed by mapping binary states ``{0,1}`` to spin values ``{-1,+1}`` internally.
    - Brute-force methods such as ``get_all_energies`` scale as O(2^n) and are intended only for
      small systems.
    """

    def __init__(self, n: int, couplings: List[dimod.BinaryPolynomial] = [], name: str = None, cost_function_signs: list = [-1, -1], model_type: str = "ising"):
        self.n = n
        self.n_spins = n
        self.couplings = couplings
        self.name = name
        self.alpha = self.calculate_alpha(n, couplings)
        self.lowest_energy = None

        self.normalised_couplings = [
            dimod.BinaryPolynomial({k: v * self.alpha for k, v in poly.items()}, poly.vartype)
            for i, poly in enumerate(self.couplings)
        ]
        self.cost_function_signs = cost_function_signs

        if model_type not in ["ising", "binary"]:
            raise ValueError(f"Invalid model_type '{model_type}'. Expected 'ising' or 'binary'.")
        else:
            self.model_type = model_type

        self.initial_state = self.get_initial_states(num_initial_states=100)

    def get_initial_states(self, num_initial_states: int):
        """
        Generates a list of random initial states.

        Parameters
        ----------
        num_initial_states:
            The number of initial states to generate.

        Returns
        -------
        list:
            A list of random initial states.
        """

        init_states = []
        while len(init_states) < num_initial_states:
            state = get_random_state(self.n_spins)
            init_states.append(state)
        return init_states


    def calculate_energy(self, state, couplings: List[dimod.BinaryPolynomial], cost_function_signs):
        """
        Calculate the energy of a given state for an arbitrary-order Ising/binary model.

        Parameters
        -----------
        state : array-like (str, list, tuple, np.array)

            State configuration. Can be:
            - Binary: "011", [0,1,1], (0,1,1), etc.
            - Spin: [-1,1,1], (-1,1,1), etc.

        couplings : list of numpy arrays
        
        List of coupling tensors where:
            - 1D arrays represent linear terms (h_i)
            - 2D arrays represent quadratic terms (J_ij)
            - 3D arrays represent cubic terms, etc.

        Returns
        -------
        float : Total energy of the state
        """

        if isinstance(state, str):
            state = np.array([int(bit) for bit in state])
        if isinstance(state, (list, tuple, np.ndarray)):
            state = np.array(state)
            if state.dtype == np.bool_:
                state = state.astype(int)
        else:
            raise TypeError(f"State must be a string, list, tuple, or numpy array, but got {type(state)}")

        if self.model_type == "binary":
            # Check if strictly binary using bitwise mask
            if np.any(state & ~1): 
                # Fallback check for strictly spin (-1, 1) using absolute value
                if np.any(np.abs(state) != 1):
                    raise ValueError("Spin configuration must be in spin (-1/+1) or binary (0/1) format.")
                # Map Spin -> Binary
                state = (1 - state) >> 1 
                
        elif self.model_type == "ising":
            # Check if strictly spin
            if np.any(np.abs(state) != 1):
                # Fallback check for strictly binary
                if np.any(state & ~1):
                    raise ValueError("Spin configuration must be in binary (0/1) or spin (-1/+1) format.")
                # Map Binary -> Spin (Shift left by 1 is 2*x, minus 1)
                state = (state << 1) - 1

        state_dict = {i: int(val) for i, val in enumerate(state)}
        total_energy = 0.0
        for term_index, poly in enumerate(couplings):
            total_energy += cost_function_signs[term_index] * poly.energy(state_dict)

        return total_energy

    def get_subgroup_couplings(self, subgroup: List[int], current_state: str, coupling_weights: List[float] = None):
        """
        Calculates local couplings for a subgroup of spins.

        Spins outside the specified subgroup are treated as frozen constants, and their
        values are absorbed into the couplings of the subgroup. This is useful for
        calculating local effective Hamiltonians.

        Parameters
        ----------
        subgroup : List[int]
            A list of indices of the spins in the subgroup.
        current_state : str
            The current state of the full system, as a binary string.
        
        coupling_weights : List[float], optional
            Optional weights for the couplings. This can be used to effectively
            remove certain couplings from the proposal (e.g., constraints) by setting
            their weight to 0. It is important not to normalize the couplings after
            applying these weights.

        Returns
        -------
        List[np.ndarray]
            A new list of coupling tensors for the subgroup.

        Notes
        -----
            It might seem off to do the reweighting at this point, but here are reasons why I [SF] did it!
            Basically, when you get the subgroup couplings, the terms jumble up, so the returned local
            coupling list necessarily does not respect the order etc. of the different terms in the original.
        """
        
        
        if coupling_weights is None:
            coupling_weights = [1.0] * len(self.normalised_couplings)

        subgroup_set = set(subgroup)
        g_to_l = {g_idx: l_idx for l_idx, g_idx in enumerate(subgroup)}

        if self.model_type == "ising":
            state_vals = {i: 1 if b == "1" else -1 for i, b in enumerate(current_state)}
        else:
            state_vals = {i: int(b) for i, b in enumerate(current_state)}

        new_couplings = []

        for couplings_index, poly in enumerate(self.normalised_couplings):
            weight = coupling_weights[couplings_index]
            new_dict = defaultdict(float)
            
            for indices, coeff in poly.items():
                effective_coeff = coeff * weight
                in_group = []
                multiplier = 1.0
                
                for i in indices:
                    if i in subgroup_set:
                        in_group.append(g_to_l[i])
                    else:
                        multiplier *= state_vals[i]
                        
                effective_coeff *= multiplier
                
                if in_group:
                    local_indices = tuple(sorted(in_group))
                    new_dict[local_indices] += effective_coeff
                else:
                    new_dict[()] += effective_coeff

            clean_dict = {k: v for k, v in new_dict.items() if v != 0}
            new_couplings.append(dimod.BinaryPolynomial(clean_dict, poly.vartype))

        return new_couplings

    def calculate_alpha(self, n: int, couplings: list = None, eps: float = 1e-15) -> np.ndarray:
        """
        Compute alpha = sqrt(n) / sqrt(sum of squares of UNIQUE coupling coefficients),
        assuming coupling tensors are symmetric representations.

        Any non-symmetric 2-body input raises ValueError.

        Parameters
        ----------
        n : int
            Number of spins.
        couplings : list[np.ndarray], optional
            Couplings to use. Defaults to self.couplings if not provided.
        eps : float
            Small threshold to avoid division by zero.

        Returns
        -------
        np.ndarray :
            An array of normalising factors for each term in the couplings list.
        """
        if couplings is None:
            couplings = self.couplings

        norm_sq_arr = np.zeros(len(couplings))
        for T_ind, poly in enumerate(couplings):
            norm_sq = sum(float(c)**2 for c in poly.values())
            norm_sq_arr[T_ind] = norm_sq

        norm_sq_tot = np.sum(norm_sq_arr)
        if norm_sq_tot < eps:
            print("Warning: Couplings are all zero (or very close to zero). Returning alpha=1 to avoid division by zero.")
            print("Input of a zero-coupling may be intended to sample Uniform distributions, however if this is not your intention, please check your couplings.")
            return np.ones(len(couplings))

        return np.sqrt(n) / np.sqrt(norm_sq_tot)

    def get_energy(self, state: str) -> float:
        """
        Returns the energy of a given state
        """
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        energy = self.calculate_energy(state, self.couplings, self.cost_function_signs)
        return energy

    def get_all_energies(self) -> np.ndarray:

        """
        Calculate the energies for all possible spin states.
        This method generates all possible spin states for the
        system and calculates the energy for each state.

        Returns
        -------
        np.ndarray :
            An array containing the energies of all possible spin states.
        """
        

        self.S = ["".join(i) for i in itertools.product("01", repeat=self.n)]
        all_energies = np.zeros(len(self.S))
        for state in self.S:
            all_energies[int(state, 2)] = self.calculate_energy(state, self.couplings, self.cost_function_signs)
        return all_energies

    def get_lowest_energies(self, num_states: int, return_configurations: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the lowest energy states and their degeneracies.
        This method computes all possible energies and then finds the specified number
        of lowest energy states along with their degeneracies.

        Parameters
        ----------
        num_states (int): The number of lowest energy states to retrieve.
        return_configurations (bool): Whether to also return the corresponding configurations of the lowest energy states. Defaults to False.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - The first array contains the lowest energy values.
            - The second array contains the degeneracies of the corresponding energy values.

        Notes
        -----
            This method is intended for small instances due to its brute-force nature, which is extremely memory intensive and slow.
        """
        all_energies = self.get_all_energies()
        lowest_energies, lowest_energy_degeneracy = self.find_lowest_values(all_energies, num_values=num_states)
        if return_configurations:
            lowest_configs = []
            for energy in lowest_energies:
                configs = [self.S[i] for i, e in enumerate(all_energies) if e == energy]
                lowest_configs.append(configs)
            self.lowest_energy = lowest_energies[0]
            return lowest_energies, lowest_energy_degeneracy, lowest_configs
        else:
            self.lowest_energy = lowest_energies[0]
            return lowest_energies, lowest_energy_degeneracy

    def find_lowest_values(self, arr: np.ndarray, num_values: int = 5):
        """
        Find the lowest unique values in an array and their degeneracies.

        Parameters
        ----------
        arr (np.ndarray): The input array from which to find the lowest values.
        num_values (int, optional): The number of lowest unique values to find. Defaults to 5.

        Returns
        -------
        Tuple[lowest_values (np.ndarray), degeneracy (np.ndarray)]
            - lowest_values (np.ndarray): The lowest unique values in the array.
            - degeneracy (np.ndarray): The counts of each of the lowest unique values.
        """
        unique_values, counts = np.unique(arr, return_counts=True)
        sorted_indices = np.argsort(unique_values)
        unique_values_sorted = unique_values[sorted_indices]
        counts_sorted = counts[sorted_indices]
        lowest_values = unique_values_sorted[:num_values]
        degeneracy = counts_sorted[:num_values]
        return lowest_values, degeneracy

    def get_lowest_energy(self):
        """
        Calculate and return the lowest energy from all possible energies.
        This method uses a brute force approach to find the lowest energy,
        making it extremely memory intensive and slow. It is recommended
        to use this method only for small instances.

        Returns
        -------
        float: The lowest energy value.

        Notes
        -----
        If the lowest energy has already been calculated and stored
        in `self.lowest_energy`, it will return that value directly
        to save computation time.
        """

        if self.lowest_energy is not None:
            return self.lowest_energy
        else:
            all_energies = self.get_all_energies()

        lowest_energy = np.min(all_energies)
        self.lowest_energy = lowest_energy
        return lowest_energy

    def get_boltzmann_factor(self, state: str, beta: float = 1.0) -> float:
        """
        Get un-normalised boltzmann probability of a given state

        Parameters
        ----------
        state : str
            configuration of spins for which probability is to be calculated
        beta : float
            inverse temperature (1/T) at which the probability is to be calculated.

        Returns
        -------
            float corresponding to the un-normalised boltzmann probability of the given state.
        """

        E = self.get_energy(state)
        boltzmann_factor = np.exp(-1 * beta * E, dtype=np.longdouble)
        return boltzmann_factor

    def get_boltzmann_factor_from_energy(self, E, beta: float = 1.0) -> float:
        """
        Get un-normalized Boltzmann probability for a given energy.

        Parameters
        ----------
        E : float
            Energy for which the Boltzmann factor is to be calculated.
        beta : float
            Inverse temperature (1/T) at which the probability is to be calculated.

        Returns
        -------
        float:
            The un-normalized Boltzmann probability for the given energy.
        """
        return np.exp(-1 * beta * E, dtype=np.longdouble)
