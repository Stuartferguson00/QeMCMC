import itertools
import typing
from typing import List
import numpy as np


class EnergyModel:
    """
    Base class for energy models. Initializes with a couplings list.
    """

    def __init__(
        self,
        n: int,
        couplings: List[np.ndarray] = [],
        name: str = None,
        alpha: float = 1.0,
        cost_function_signs: list = [-1, -1],
    ):
        self.n = n
        self.n_spins = n
        self.couplings = couplings
        self.name = name
        self.alpha = self.calculate_alpha(n, couplings)
        self.cost_function_signs = cost_function_signs
        self.initial_state = []

        for i in range(100):
            self.initial_state.append("".join(str(i) for i in np.random.randint(0, 2, self.n, dtype=int)))

    def calculate_energy(self, state, couplings, spin_type="binary", sign=1):
        """
        Calculate the energy of a given state for an arbitrary-order Ising/QUBO model.

        Parameters:
        -----------
            state : array-like (str, list, tuple, np.array)
                State configuration. Can be:
                - Binary: "011", [0,1,1], (0,1,1), etc. (uses values 0 and 1 directly)
                - Spin: [-1,1,1], (-1,1,1), etc. (uses values -1 and +1 directly)

            couplings : list of numpy arrays
                List of coupling tensors where:
                - 1D arrays represent linear terms (h_i)
                - 2D arrays represent quadratic terms (J_ij)
                - 3D arrays represent cubic terms, etc.

            spin_type : str, optional (default='binary')
                - 'binary': state uses 0/1 values
                - 'spin': state uses -1/+1 values
                Note: The actual values in 'state' are used directly in the calculations

        Returns:
        --------
            float : Total energy of the state
        """
        if isinstance(state, str):
            state = np.array([int(bit) for bit in state])
        else:
            state = np.array(state)

        if spin_type == "binary":
            state = 2 * state - 1

        total_energy = 0.0
        for coupling in couplings:
            coupling = np.array(coupling)
            order = coupling.ndim

            if order == 1:
                total_energy += np.dot(coupling, state)
            elif order == 2:
                total_energy += np.einsum("ij, i, j->", coupling, state, state)
            else:
                # General case for any order >=3 (cubic, quartic etc.)
                indices = "".join(chr(97 + i) for i in range(order))  # 'abc...', 'ijkl...'
                einsum_str = f"{indices}," + ",".join([indices[i] for i in range(order)]) + "->"
                total_energy += np.einsum(einsum_str, coupling, *([state] * order))

        return sign * total_energy

    def get_subgroup_couplings(self, subgroup: List[int], current_state: str):
        """
        Calculates local couplings for a subgroup.
        Spins outside the group are treated as frozen constants.
        """
        n_sub = len(subgroup)
        subgroup_set = set(subgroup)
        g_to_l = {g_idx: l_idx for l_idx, g_idx in enumerate(subgroup)}

        # Map bitstring '0'/'1' to spin values -1/+1
        state_vals = np.array([1 if b == "1" else -1 for b in current_state])
        max_order = max(c.ndim for c in self.couplings)
        new_couplings = [np.zeros((n_sub,) * d) for d in range(1, max_order + 1)]

        for coupling in self.couplings:
            for indices in np.ndindex(coupling.shape):
                coeff = coupling[indices]
                if coeff == 0 or len(set(indices)) != len(indices):
                    continue

                in_group = [i for i in indices if i in subgroup_set]
                out_group = [i for i in indices if i not in subgroup_set]

                # Multiply coefficient by values of fixed spins outside the subgroup
                multiplier = np.prod(state_vals[out_group])
                effective_coeff = coeff * multiplier

                if in_group:
                    new_order = len(in_group)
                    local_indices = tuple(g_to_l[i] for i in in_group)
                    new_couplings[new_order - 1][local_indices] += effective_coeff

        return new_couplings

    def calculate_alpha(self, n, couplings, eps: float = 1e-15) -> float:
        """
        Compute alpha = sqrt(n) / sqrt(sum of squares of UNIQUE coupling coefficients),
        assuming coupling tensors are symmetric representations.

        Any non-symmetric 2-body input raises ValueError.

        Parameters
        ----------
        couplings : list[np.ndarray] | None
            Couplings to use. Defaults to self.couplings.
        eps : float
            Small threshold to avoid division by zero.

        Returns
        -------
        float : alpha
        """
        if couplings is None:
            couplings = self.couplings

        norm_sq = 0.0

        for T in couplings:
            T = np.asarray(T)
            order = T.ndim

            if order == 0:
                continue

            # 1-body: h_i
            if order == 1:
                if T.shape != (n,):
                    raise ValueError(f"1-body tensor has shape {T.shape}, expected ({n},)")
                for i in range(n):
                    c = float(T[i])
                    norm_sq += c * c
                continue

            # 2-body: symmetric J
            if order == 2:
                if T.shape != (n, n):
                    raise ValueError(f"2-body tensor has shape {T.shape}, expected ({n},{n})")

                # Enforce symmetry (rejects pure upper/lower triangular)
                if not np.allclose(T, T.T):
                    raise ValueError("Non-symmetric J provided. This alpha function only accepts symmetric J.")

                # Count each interaction once: i<j
                for i in range(n):
                    for j in range(i + 1, n):
                        c = float(T[i, j])
                        if c != 0.0:
                            norm_sq += c * c
                continue

            # Order >= 3: count each unordered interaction once using i1<i2<...<ik
            if T.shape != (n,) * order:
                raise ValueError(f"{order}-body tensor has shape {T.shape}, expected {(n,) * order}")

            for comb in itertools.combinations(range(n), order):
                c = float(T[comb])
                if c != 0.0:
                    norm_sq += c * c

        if norm_sq < eps:
            raise ValueError("Cannot compute alpha: no nonzero (non-constant) couplings found.")

        return np.sqrt(n / norm_sq)

    def get_energy(self, state: str) -> float:
        """
        Returns the energy of a given state
        """
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        energy = self.calculate_energy(state, self.couplings)
        return energy

    def get_all_energies(self) -> np.ndarray:
        """
        Calculate the energies for all possible spin states.
        This method generates all possible spin states for the
        system and calculates the energy for each state.

        Returns
        -------
            np.ndarray: An array containing the energies of all possible spin states.
        """
        self.S = ["".join(i) for i in itertools.product("01", repeat=self.n)]
        all_energies = np.zeros(len(self.S))
        for state in self.S:
            all_energies[int(state, 2)] = self.calculate_energy(state, self.couplings)
        return all_energies

    def get_lowest_energies(self, num_states: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the lowest energy states and their degeneracies.
        This method computes all possible energies and then finds the specified number
        of lowest energy states along with their degeneracies. Note that this method
        is intended for small instances due to its brute-force nature, which is extremely
        memory intensive and slow.
        Args:
            num_states (int): The number of lowest energy states to retrieve.
        Returns:
            Two numpy arrays:
                - The first array contains the lowest energy values.
                - The second array contains the degeneracies of the corresponding energy values.
        """
        # only to be used for small instances, it is just brute force so extremely memory intensive and slow
        all_energies = self.get_all_energies()

        # very slow (sorts whole array)
        self.lowest_energies, self.lowest_energy_degeneracy = self.find_lowest_values(all_energies, num_values=num_states)

        return self.lowest_energies, self.lowest_energy_degeneracy

    def find_lowest_values(self, arr: np.ndarray, num_values: int = 5):
        """
        Find the lowest unique values in an array and their degeneracies.

        Args:
            arr (np.ndarray): The input array from which to find the lowest values.
            num_values (int, optional): The number of lowest unique values to find. Defaults to 5.

        Returns:
        tuple: A tuple containing two numpy arrays:
            - lowest_values (np.ndarray): The lowest unique values in the array.
            - degeneracy (np.ndarray): The counts of each of the lowest unique values.
        """
        # Count the occurrences of each value
        unique_values, counts = np.unique(arr, return_counts=True)
        # Sort the unique values and counts by value
        sorted_indices = np.argsort(unique_values)
        unique_values_sorted = unique_values[sorted_indices]
        counts_sorted = counts[sorted_indices]
        # Find the first num_values
        lowest_values = unique_values_sorted[:num_values]
        degeneracy = counts_sorted[:num_values]
        return lowest_values, degeneracy

    def get_lowest_energy(self):
        """
        Calculate and return the lowest energy from all possible energies.
        This method uses a brute force approach to find the lowest energy,
        making it extremely memory intensive and slow. It is recommended
        to use this method only for small instances.
        Returns:
            float: The lowest energy value.
        Notes:
            If the lowest energy has already been calculated and stored
            in `self.lowest_energy`, it will return that value directly
            to save computation time.
        """

        # Only to be used for small instances, it is just brute force so extremely memory intensive and slow
        if self.lowest_energy is not None:
            return self.lowest_energy
        else:
            all_energies = self.get_all_energies()

        lowest_energy = np.min(all_energies)

        return lowest_energy

    def get_boltzmann_factor(self, state: str, beta: float = 1.0) -> float:
        """
        Get un-normalised boltzmann probability of a given state

        Args:
            state (str): configuration of spins for which probability is to be calculated
            beta (float): inverse temperature (1/T) at which the probability is to be calculated.

        Returns:
            float corresponding to the un-normalised boltzmann probability of the given state.
        """
        E = self.get_energy(state)
        r = np.exp(-1 * beta * E, dtype=np.longdouble)

        return r

    def get_boltzmann_factor_from_energy(self, E, beta: float = 1.0) -> float:
        """
        Get un-normalized Boltzmann probability for a given energy.

        Args:
            E (float): Energy for which the Boltzmann factor is to be calculated.
            beta (float): Inverse temperature (1/T) at which the probability is to be calculated.

        Returns:
            float: The un-normalized Boltzmann probability for the given energy.
        """

        return np.exp(-1 * beta * E, dtype=np.longdouble)
