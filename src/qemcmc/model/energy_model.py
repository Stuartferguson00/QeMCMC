import itertools
import typing
from typing import List
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector


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
        self.alpha = alpha
        self.cost_function_signs = cost_function_signs
        self.initial_state = []

        for i in range(100):
            self.initial_state.append("".join(str(i) for i in np.random.randint(0, 2, self.n, dtype=int)))

    def calc_an_energy(self, state):
        return self.calculate_energy(state, self.couplings)

    def get_hamiltonian(self, spin_type="binary", sign=1):
        H_sparse = self.couplings_to_sparse_pauli(self.n, self.couplings, sign)
        return H_sparse.to_matrix().real

    def couplings_to_sparse_pauli(self, n, couplings, sign=-1):
        """
        This function basically maps a classical energy model into a quantum operator (hamiltonian)
        It converts an arbitrary-order list of couplings directly to SparsePauliOp.
        Never constructs the full 2^n matrix!

        Parameters:
            n: number of qubits
            couplings: list of coupling tensors (1D=linear, 2D=quadratic, 3D=cubic, etc.)
            sign: overall sign

        Returns:
            SparsePauliOp
        """
        pauli_list = []

        for coupling in couplings:
            coupling = np.array(coupling)
            order = coupling.ndim

            # Sign factor: (-1)^order to convert from σ = -Z convention
            spin_sign = (-1) ** order

            if order == 1:
                # Linear terms: h_i Z_i
                for i in range(n):
                    if coupling[i] != 0:
                        pauli_str = ["I"] * n
                        pauli_str[i] = "Z"  # No reversal - match user's big-endian
                        # Negate because σ = -Z (spin convention flip)
                        pauli_list.append(("".join(pauli_str), -sign * coupling[i]))

            elif order == 2:
                # Quadratic terms: J_ij Z_i Z_j
                for i in range(n):
                    for j in range(n):
                        if i != j and coupling[i, j] != 0:
                            pauli_str = ["I"] * n
                            pauli_str[i] = "Z"  # No reversal
                            pauli_str[j] = "Z"
                            # No sign flip needed: σ_i σ_j = (-Z_i)(-Z_j) = Z_i Z_j
                            pauli_list.append(("".join(pauli_str), sign * coupling[i, j]))

            elif order >= 3:
                # Higher-order terms: K_ijk... Z_i Z_j Z_k ...
                for indices in np.ndindex(coupling.shape):
                    if coupling[indices] != 0 and len(set(indices)) == len(indices):
                        pauli_str = ["I"] * n
                        for idx in indices:
                            pauli_str[idx] = "Z"  # Big-endian: no reversal
                        pauli_list.append(("".join(pauli_str), sign * spin_sign * coupling[indices]))

        # return SparsePauliOp.from_list(pauli_list).simplify()
        return SparsePauliOp.from_list(pauli_list)

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

    def get_energy(self, state: str) -> float:
        """
        Returns the energy of a given state
        """
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        energy = self.calc_an_energy(state)
        return energy

    def get_all_energies(self) -> np.ndarray:
        """
        Calculate the energies for all possible spin states.
        This method generates all possible spin states for the system, calculates the energy for each state,
        and returns an array of these energies.
        Returns:
            np.ndarray: An array containing the energies of all possible spin states.
        """
        self.S = ["".join(i) for i in itertools.product("01", repeat=self.n)]
        all_energies = np.zeros(len(self.S))
        for state in self.S:
            all_energies[int(state, 2)] = self.calc_an_energy(state)
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


if __name__ == "__main__":
    # Linear coefficients (h vector)
    h = np.array([-1.0, -2.0, -3.0])

    # Quadratic coefficients (J matrix)
    # J_12 = 0.5, J_23 = -1.5
    J = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -1.5], [0.0, -1.5, 0.0]])

    # Create the list of coupling tensors
    my_couplings = [h, 0.5 * J]

    my_state = "011"

    energies = []
    energies3 = []
    energies_H = []
    energy_model = EnergyModel(n=3, couplings=my_couplings)
    H = energy_model.get_hamiltonian(spin_type="binary", sign=-1)

    H_sparse = energy_model.couplings_to_sparse_pauli(3, my_couplings, sign=-1)

    # All possible spin states
    spin_states = [
        [-1, -1, -1],  # "000"
        [-1, -1, 1],  # "001"
        [-1, 1, -1],  # "010"
        [-1, 1, 1],  # "011"
        [1, -1, -1],  # "100"
        [1, -1, 1],  # "101"
        [1, 1, -1],  # "110"
        [1, 1, 1],  # "111"
    ]

    # Corresponding binary strings (for comparison and Statevector)
    binary_states = ["000", "001", "010", "011", "100", "101", "110", "111"]

    my_couplings_spin = [h, 0.5 * J]  # Keep same for consistency

    energy_model = EnergyModel(n=3, couplings=my_couplings_spin)
    H_sparse = energy_model.couplings_to_sparse_pauli(3, my_couplings_spin, sign=-1)

    energies_spin = []
    energies_binary = []
    energies_H_sparse = []

    for spin_state, binary_state in zip(spin_states, binary_states):
        print(f"\nSpin state: {spin_state}  (binary: {binary_state})")
        print("-" * 40)

        # Method 1: calculate_energy with spin_type="spin"
        energy_spin = energy_model.calculate_energy(spin_state, my_couplings_spin, spin_type="spin", sign=-1)
        print(f"Energy (spin input):   {energy_spin:.4f}")
        energies_spin.append(energy_spin)

        # Method 2: calculate_energy with spin_type="binary" (for comparison)
        energy_binary = energy_model.calculate_energy(binary_state, my_couplings_spin, spin_type="binary", sign=-1)
        print(f"Energy (binary input): {energy_binary:.4f}")
        energies_binary.append(energy_binary)

        # Method 3: SparsePauliOp (always uses computational basis internally)
        # Convert spin to binary for Statevector: -1 -> "0", +1 -> "1"
        sv_label = "".join("0" if s == -1 else "1" for s in spin_state)
        sv = Statevector.from_label(sv_label)  # Reverse for endianness
        energy_pauli = sv.expectation_value(H_sparse).real
        print(f"Energy (SparsePauliOp): {energy_pauli:.4f}")
        energies_H_sparse.append(energy_pauli)

        # Check match
        if np.isclose(energy_spin, energy_binary) and np.isclose(energy_spin, energy_pauli):
            print("✓ All methods match!")
        else:
            print("✗ MISMATCH!")

        print("Calculating energy for state:", binary_state)

        energy3 = energy_model.calculate_energy(binary_state, my_couplings, sign=-1)
        print("Energy using new formula:", energy3)
        energies3.append(energy3)

        state_index = int(binary_state, 2)  # "010" -> 2
        energy_H = H[state_index, state_index]
        print("Energy using Hamiltonian:", energy_H)
        energies_H.append(energy_H)

        sv = Statevector.from_label(binary_state)
        energy = sv.expectation_value(H_sparse).real
        print(f"State {binary_state}: E = {energy:.4f}")

        print("\n\n")
