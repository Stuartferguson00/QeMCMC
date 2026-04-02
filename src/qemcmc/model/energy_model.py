import itertools
import typing
from typing import List
import numpy as np
import dimod
import math
from tqdm import tqdm
from qemcmc.utils.helpers import get_random_state

class EnergyModel:
    """
    A base class for energy models for a classical energy function over n spins,
    defined by arbitrary-order coupling tensors.

    Parameters
    ----------
    n:
        Number of spins
    couplings:
        List of numpy arrays representing interaction tensors. A 1D array encodes linear terms,
        a 2D array encodes pairwise terms (expected symmetric), and higher-rank tensors encode
        higher-order interactions.
    name:
        Optional label for the model (used in plotting / logging).
    cost_function_signs:
        Sign convention(s) used by downstream components (e.g. proposal/acceptance conventions).
    model_type: str
        Type of model, either 'ising' or 'qubo'. This determines how the binary states are interpreted
        and how the energy is calculated. 'ising' models use spin values {-1, +1}, while 'qubo'
        models use binary values {0, 1}.
    Notes
    -----
    - Energies are computed by mapping binary states ``{0,1}`` to spin values ``{-1,+1}`` internally.
    - Brute-force methods such as ``get_all_energies`` scale as O(2^n) and are intended only for
      small systems.
    """

    def __init__(
        self,
        n: int,
        couplings: List[np.ndarray] = [],
        name: str = None,
        cost_function_signs: list = [-1, -1],
        model_type: str = "ising"
    ):
        self.n = n
        self.n_spins = n
        self.couplings = couplings
        self.name = name
        self.alphas = self.calculate_alpha(n, couplings)
        self.normalised_couplings = [self.couplings[i] * self.alphas[i] for i in range(len(self.couplings))]
        self.cost_function_signs = cost_function_signs

        if model_type not in ["ising", "binary"]:
            raise ValueError(f"Invalid model_type '{model_type}'. Expected 'ising' or 'binary'.")
        else:
            self.model_type = model_type

        #for i in range(100):
        #    self.initial_state.append(get_random_state(self.n))
        self.initial_state = self.get_initial_states(num_initial_states=100)

    def get_initial_states(self, num_initial_states:int):
        init_states = []
        while len(init_states) < num_initial_states:
            state = get_random_state(self.n_spins)
            init_states.append(state)
        return init_states

    def get_ground_state(self, num_reads=100, num_batches=10):
        """
        Finds an approximate ground state using Simulated Annealing.
        """
        h, J = self.couplings

        h_dict = {i: h[i] for i in range(self.n_spins)}
        J_dict = {(i, j): J[i, j] for i in range(self.n_spins) for j in range(i + 1, self.n_spins)}

        bqm = dimod.BinaryQuadraticModel.from_ising(h_dict, J_dict)

        sampler = dimod.SimulatedAnnealingSampler()
        best_energy = float("inf")

        reads_per_batch = max(1, num_reads // num_batches)

        for _ in tqdm(range(num_batches), desc=f"Annealing ({num_reads} total reads)"):
            response = sampler.sample(bqm, num_reads=reads_per_batch)

            current_best = response.first
            if current_best.energy < best_energy:
                best_energy = current_best.energy

        print("\n--- Simulated Annealing Results ---")
        print(f"Lowest Energy Found: {best_energy:.4f}")

        return best_energy

    def calculate_energy(self, state, couplings, cost_function_signs):
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

        #print("state before:", state)
        if self.model_type == "binary":
            if not np.all(np.isin(state, [0, 1])):
                # If not already in binary format, try interpreting as spin values and converting
                if not np.all(np.isin(state, [-1, 1])):
                    # not in spin format either?
                    raise ValueError("Spin configuration must be in spin (-1/+1) or binary (0/1) format, but got values outside these sets.")
                else:
                    # Convert to binary
                    state = np.array([(spin + 1) // 2 for spin in state])
        elif self.model_type == "ising":
            if not np.all(np.isin(state, [-1, 1])):
                # If not already in spin format, try interpreting as binary and converting
                if not np.all(np.isin(state, [0, 1])):
                    # not in binary format either?
                    raise ValueError("Spin configuration must be in binary (0/1) or spin (-1/+1) format, but got values outside these sets.")
                else:
                    # Convert to spin
                    state = np.array([2 * int(bit) - 1 for bit in state])
        
        #print("state after:", state)
        total_energy = 0.0
        for term_index, coupling in enumerate(couplings):
            coupling = np.array(coupling)
            order = coupling.ndim

            if order == 1:
                total_energy += cost_function_signs[term_index] * np.dot(coupling, state)
            elif order == 2:
                total_energy += cost_function_signs[term_index] * 0.5 * np.einsum("ij, i, j->", coupling, state, state)
            else:
                # General case for any order >=3 (cubic, quartic etc.)
                indices = "".join(chr(97 + i) for i in range(order))  # 'abc...', 'ijkl...'
                einsum_str = f"{indices}," + ",".join([indices[i] for i in range(order)]) + "->"
                coefficient = 1.0 / math.factorial(order)
                total_energy += cost_function_signs[term_index] * coefficient * np.einsum(einsum_str, coupling, *([state] * order))

        return total_energy

    def get_subgroup_couplings(self, subgroup: List[int], current_state: str, coupling_weights: List[float] = None):
        """
        Calculates local couplings for a subgroup.
        Spins outside the group are treated as frozen constants.
        coupling_weights adds in the option to weight the couplings, which can be used to effectively remove certain couplings from the proposal (e.g. constraints) by setting their weight to 0. 
        Very important to NOT normalise after this ste though. 
        
        It might seem off to do the reweighting at this point, but here are reasons why I [SF] did it! Basically, when you get the subgroup couplings, the terms jumble up, so the returned local coupling list necessarily does not respect the order etc. of the different terms in the original.
        """
        if coupling_weights is None:
            coupling_weights = [1.0] * len(subgroup)

        n_sub = len(subgroup)
        subgroup_set = set(subgroup)
        g_to_l = {g_idx: l_idx for l_idx, g_idx in enumerate(subgroup)}

        # Map bitstring '0'/'1' to spin values -1/+1
        state_vals = np.array([1 if b == "1" else -1 for b in current_state])
        max_order = max(c.ndim for c in self.normalised_couplings)
        new_couplings = [np.zeros((n_sub,) * d) for d in range(1, max_order + 1)]

        for couplings_index, coupling in enumerate(self.normalised_couplings):
            coupling = np.array(coupling) * coupling_weights[couplings_index]
            # only loop over elements that actually exist (non-zero)
            non_zero_indices = np.transpose(np.nonzero(coupling))
            
            for indices in non_zero_indices:
                indices = tuple(indices)
                coeff = coupling[indices]

                if len(set(indices)) != len(indices):
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

    def calculate_alpha(self, n: int, couplings, eps: float = 1e-15) -> float:
        """
        Compute alpha = sqrt(n) / sqrt(sum of squares of UNIQUE coupling coefficients),
        assuming coupling tensors are symmetric representations.

        Any non-symmetric 2-body input raises ValueError.

        Parameters
        ----------
        n : int
            Number of spins.
        couplings : list[np.ndarray] | None
            Couplings to use. Defaults to self.couplings.
        eps : float
            Small threshold to avoid division by zero.

        Returns
        -------
        float : alpha
            normalising factor for each term in the coupligs list
        """
        if couplings is None:
            couplings = self.couplings

        norm_sq_arr = np.zeros(len(couplings))
        for T_ind, T in enumerate(couplings):
            norm_sq = 0.0
            T = np.asarray(T)
            order = T.ndim

            if order == 0:
                pass

            # 1-body: h_i
            if order == 1:
                if T.shape != (n,):
                    raise ValueError(f"1-body tensor has shape {T.shape}, expected ({n},)")
                for i in range(n):
                    c = float(T[i])
                    norm_sq += c * c

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
                

            # Order >= 3: count each unordered interaction once using i1<i2<...<ik
            if T.shape != (n,) * order:
                raise ValueError(f"{order}-body tensor has shape {T.shape}, expected {(n,) * order}")

            for comb in itertools.combinations(range(n), order):
                c = float(T[comb])
                if c != 0.0:
                    norm_sq += c * c
            norm_sq_arr[T_ind] = norm_sq

        norm_sq_tot = np.sum(norm_sq_arr)
        if norm_sq_tot < eps:
            raise ValueError("Cannot compute alpha: no nonzero (non-constant) couplings found.")

        return np.sqrt(n / norm_sq_arr)

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
            np.ndarray: An array containing the energies of all possible spin states.
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
        of lowest energy states along with their degeneracies. Note that this method
        is intended for small instances due to its brute-force nature, which is extremely
        memory intensive and slow.
        Args:
            num_states (int): The number of lowest energy states to retrieve.
            return_configurations (bool): Whether to also return the corresponding configurations of the lowest energy states. Defaults to False.
        Returns:
            Two numpy arrays:
                - The first array contains the lowest energy values.
                - The second array contains the degeneracies of the corresponding energy values.
        """
        # only to be used for small instances, it is just brute force so extremely memory intensive and slow
        all_energies = self.get_all_energies()

        # very slow (sorts whole array)
        self.lowest_energies, self.lowest_energy_degeneracy = self.find_lowest_values(all_energies, num_values=num_states)
        if return_configurations:
            lowest_configs = []
            for energy in self.lowest_energies:
                configs = [self.S[i] for i, e in enumerate(all_energies) if e == energy]
                lowest_configs.append(configs)
            return self.lowest_energies, self.lowest_energy_degeneracy, lowest_configs
        else:
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
    
class ConstraintModel(EnergyModel):
    """
    A subclass of EnergyModel that incorporates a constraint function to define valid configurations.
    The constraint function takes a state as input and returns True if the state is valid (satisfies the constraint) and False otherwise. The energy of invalid states is set to infinity, effectively excluding them from the Boltzmann distribution.

    Parameters
    ----------
    n : int
        Number of spins in the model.
    constraint_func : callable
        A function that takes a state (string representation of spin configuration) and returns True if the state satisfies the constraint, and False otherwise.
    constraint_couplings : list
        List of coupling tensors (numpy arrays) defining the constraint.
    couplings : list
        List of coupling tensors (numpy arrays) defining the energy function.
    name:
        Optional label for the model (used in plotting / logging).
    constraint_signs:
        Sign convention(s) for the constraint couplings.
    cost_function_signs:
        Sign convention(s) used by downstream components (e.g. proposal/acceptance conventions).
    model_type: str
        Type of model, either 'ising' or 'qubo'. This determines how the binary states are interpreted
        and how the energy is calculated. 'ising' models use spin values {-1, +1}, while 'qubo'
        models use binary values {0, 1}.

        
    name : str, optional
        An optional name for the model.

    Notes
    -----
    - The energy of any state that does not satisfy the constraint is set to infinity, which means such states will have zero probability in the Boltzmann distribution.
    - This class can be used to model systems with hard constraints on the configurations, such as certain combinatorial optimization problems or physical systems with forbidden states.
    """

    def __init__(self, n: int, constraint_couplings: list, constraint_signs: list, couplings: list, constraint_func: callable,  **kwargs):
        if constraint_func is not None:
            if not callable(constraint_func):
                raise ValueError("constraint_func must be a callable function that takes a state as input and returns True/False.")
            self.constraint_func = constraint_func
        else: # constraint_func is None
            raise  ValueError("No constraint function provided.")
        self.get_initial_states = self.get_initial_states_constraint
        super().__init__(n=n, couplings=couplings, **kwargs)
        self.constraint_couplings = constraint_couplings
        self.constraint_signs = constraint_signs
        
            
        self.constraint_coupling_alphas = self.calculate_alpha(n, constraint_couplings)
        self.normalised_couplings = [self.couplings[i] * self.alphas[i] for i in range(len(self.couplings))] + [self.constraint_couplings[i] * self.constraint_coupling_alphas[i] for i in range(len(self.constraint_couplings))]
        self.total_couplings = self.couplings + self.constraint_couplings
        

        self.initial_state = self.get_initial_states(num_initial_states=20)

    def get_initial_states_constraint(self, num_initial_states:int):
        init_states = []
        counter = 0
        while len(init_states) < num_initial_states:
            state = get_random_state(self.n_spins)
            if self.constraint_func(state):
                init_states.append(state)
            counter += 1
            if counter > 1000:
                print(f"Could not find enough valid initial states satisfying the constraint. Please provide some manually if you want more than the ones found here. Found {len(init_states)} valid states after 1000 attempts.")
                break
        return init_states



    def get_constraint_energy(self, state: str) -> float:
        """
        Calculate the energy contribution from the constraint couplings for a given state.

        Args:
            state (str): The state for which to calculate the constraint energy.

        Returns:
            float: The energy contribution from the constraint couplings for the given state.
        """
        return self.calculate_energy(state, self.constraint_couplings, self.constraint_signs)  # Assuming constraint couplings contribute positively to energy
    
    def get_total_energy(self, state: str) -> float:
        """
        Calculate the total energy of a given state, including both the regular energy and the constraint energy.

        Args:
            state (str): The state for which to calculate the total energy.

        Returns:
            float: The total energy of the given state, including contributions from both the regular couplings and the constraint couplings.
        """
        return self.get_energy(state) + self.get_constraint_energy(state)
    

    