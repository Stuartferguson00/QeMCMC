import numpy as np
from collections import Counter
from typing import Optional, List, Sequence, Tuple, Union, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


###########################################################################################
## MCMC Chain and States ##
###########################################################################################


@dataclass
class MCMCState:
    bitstring: str
    accepted: bool
    energy: float = None
    position: int = None


@dataclass(init=True)
class MCMCChain:
    def __init__(self, states: Optional[List[MCMCState]] = None, name: Optional[str] = "MCMC"):
        self.name = name

        if len(states) is None:
            self._states: List[MCMCState] = []
            self._current_state: MCMCState = None
            self._states_accepted: List[MCMCState] = []
            self.markov_chain: List[str] = []

        else:
            self._states = states
            self._current_state: MCMCState = next((s for s in self._states[::-1] if s.accepted), None)
            self._states_accepted: List[MCMCState] = [state for state in states if state.accepted]
            self.markov_chain: List[str] = self.get_list_markov_chain()

    def add_state(self, state: MCMCState):
        if state.accepted:
            self._current_state = state
            self._states_accepted.append(state)
        self.markov_chain.append(self._current_state.bitstring)
        self._states.append(state)

    @property
    def states(self):
        return self._states

    def get_accepted_energies(self):
        self.accepted_energies = []
        self.accepted_positions = []

        for state in self._states_accepted:
            self.accepted_energies.append(state.energy)
            self.accepted_positions.append(state.position)

        self.accepted_positions = np.array(self.accepted_positions)
        self.accepted_energies = np.array(self.accepted_energies)

        return self.accepted_energies, self.accepted_positions

    def get_current_energy_array(self):
        # returns the array of current energy across the entire number of hops
        # ie returns the last accepted energy
        # Useful for plotting etc

        current_energy_array = []
        for state in self._states:
            if state.accepted:
                current_energy_array.append(state.energy)
            else:
                current_energy_array.append(current_energy_array[-1])
        return np.array(current_energy_array)

    def get_pos_array(self):
        # returns the array of current pos across the entire number of hops
        # Useful for plotting etc

        pos_array = []
        for state in self._states:
            pos_array.append(state.position)
        return np.array(pos_array)

    def get_current_state_array(self):
        # returns the array of current state across the entire number of hops
        # ie returns the last accepted state
        # Useful for plotting etc

        current_state_array = []
        for state in self._states:
            if state.accepted:
                current_state_array.append(state.bitstring)
            else:
                current_state_array.append(current_state_array[-1])
        return np.array(current_state_array)

    def get_all_energies(self):
        self.energies = []
        for state in self._states:
            self.energies.append(state.energy)
        return self.energies

    @property
    def current_state(self):
        return self._current_state

    @property
    def accepted_states(self) -> List[str]:
        return [state.bitstring for state in self._states_accepted]

    def get_list_markov_chain(self) -> List[str]:
        markov_chain_in_state = [self.states[0].bitstring]
        for i in range(1, len(self.states)):
            mcmc_state = self.states[i].bitstring
            whether_accepted = self.states[i].accepted
            if whether_accepted:
                markov_chain_in_state.append(mcmc_state)
            else:
                markov_chain_in_state.append(markov_chain_in_state[i - 1])
        self.markov_chain = markov_chain_in_state
        return self.markov_chain

    def get_accepted_dict(self, normalize: bool = False, until_index: int = -1):
        if until_index != -1:
            accepted_states = self.markov_chain[:until_index]
        else:
            accepted_states = self.markov_chain

        if normalize:
            length = len(accepted_states)
            accepted_dict = Counter({s: count / length for s, count in Counter(accepted_states).items()})
        else:
            accepted_dict = Counter(accepted_states)

        return accepted_dict


def plot_chains(chains: list[MCMCChain], color: str, label: str):
    for chain in chains:
        energies = chain.get_current_energy_array()
        pos = chain.get_pos_array()
        plt.plot(pos, energies, color=color, alpha=0.1)
    avg_energy = sum(chain.get_current_energy_array() for chain in chains) / len(chains)
    plt.plot(pos, avg_energy, color=color, label=f"Average {label}")


def get_random_state(num_spins: int) -> str:
    """
    Generate a random state for a given number of spins.
    Args:
        num_spins (int): The number of spins in the system.
    Returns:
        str: A bitstring representing the random state.
    """
    # Define the size of state space
    state_space = 2 ** (num_spins)

    # Generate a random state
    next_state = np.random.randint(0, state_space, 1)[0]

    # Convert the state from integer to a bitstring
    s_prime = f"{next_state:0{num_spins}b}"
    return s_prime


def get_all_possible_states(num_spins: int) -> list:
    """
    Returns all possible binary strings of length n=num_spins

    Args:
    num_spins: n length of the bitstring
    Returns:
    possible_states= list of all possible binary strings of length num_spins
    """
    num_possible_states = 2 ** (num_spins)
    possible_states = [f"{k:0{num_spins}b}" for k in range(0, num_possible_states)]
    return possible_states


def magnetization_of_state(bitstring: str) -> float:
    """
    Args:
    bitstring: for eg: '010'
    Returns:
    magnetization for the given bitstring
    """

    if type(bitstring) is not str:
        raise TypeError("bitstring must be a string in magnetization_of_state")

    array = np.array(list(bitstring))
    num_times_one = np.count_nonzero(array == "1")
    num_times_zero = len(array) - num_times_one
    magnetization = num_times_one - num_times_zero
    n_spins = len(array)
    return magnetization / n_spins


def dict_magnetization_of_all_states(list_all_possible_states: list) -> dict:
    """
    Returns magnetization for all unique states

    Args:
    list_all_possible_states
    Returns:
    dict_magnetization={state(str): magnetization_value}
    """
    list_mag_vals = [magnetization_of_state(state) for state in list_all_possible_states]
    dict_magnetization = dict(zip(list_all_possible_states, list_mag_vals))
    # print("dict_magnetization:"); print(dict_magnetization)
    return dict_magnetization


def hamming_dist(str1, str2):
    i = 0
    count = 0
    while i < len(str1):
        if str1[i] != str2[i]:
            count += 1
        i += 1
    return count


def hamming_dist_related_counts(num_spins: int, sprime_each_iter: list, states_accepted_each_iter: list):
    dict_counts_states_hamming_dist = dict(zip(list(range(0, num_spins + 1)), [0] * (num_spins + 1)))
    ham_dist_s_and_sprime = np.array([hamming_dist(states_accepted_each_iter[j], sprime_each_iter[j + 1]) for j in range(0, len(states_accepted_each_iter) - 1)])
    for k in list(dict_counts_states_hamming_dist.keys()):
        dict_counts_states_hamming_dist[k] = np.count_nonzero(ham_dist_s_and_sprime == k)

    assert sum(list(dict_counts_states_hamming_dist.values())) == len(sprime_each_iter) - 1
    return dict_counts_states_hamming_dist


def energy_difference_related_counts(num_spins, sprime_each_iter: list, states_accepted_each_iter: list, model_in):
    energy_diff_s_and_sprime = np.array([abs(model_in.get_energy(sprime_each_iter[j]) - model_in.get_energy(states_accepted_each_iter[j + 1])) for j in range(0, len(sprime_each_iter) - 1)])
    return energy_diff_s_and_sprime


# ###########################################################################################
# ======================= Coarse Graining helper functions: ===============================
# ###########################################################################################


def validate_sub_groups(sub_groups: List[Sequence[int]], n: int) -> None:
    """
    Validate coarse-graining subgroups.

    Requirements:
      - sub_groups is a non-empty list of non-empty sequences
      - each element is an int in [0, n-1]
      - each subgroup has no duplicate indices
      - coverage: every spin 0..n-1 appears in at least one subgroup

    Raises
    ------
    ValueError message if validation fails.
    """
    if sub_groups is None:
        raise ValueError("sub_groups is None; expected a list of groups.")
    if not isinstance(sub_groups, list) or len(sub_groups) == 0:
        raise ValueError("sub_groups must be a non-empty list of groups (each group is a list of ints).")

    covered = set()

    for gi, group in enumerate(sub_groups):
        if group is None or len(group) == 0:
            raise ValueError(f"sub_groups[{gi}] is empty; each group must contain at least one spin index.")

        group_list = list(group)

        # check duplicates within a group
        if len(set(group_list)) != len(group_list):
            raise ValueError(f"sub_groups[{gi}] contains duplicate indices: {group_list}")

        for index in group_list:
            if not isinstance(index, int):
                raise ValueError(f"sub_groups[{gi}] contains non-int index {index} (type {type(index)}).")
            if index < 0 or index >= n:
                raise ValueError(f"sub_groups[{gi}] contains out-of-range index {index}; valid range is 0..{n - 1}.")
            covered.add(index)

    missing = set(range(n)) - covered
    if missing:
        raise ValueError(f"sub_groups do not cover the full system. Missing spins: {sorted(missing)}. Every spin 0..{n - 1} must appear in at least one subgroup.")
