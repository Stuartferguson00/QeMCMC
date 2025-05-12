import pytest
from qemcmc.helpers import (
    MCMCState,
    MCMCChain,
    get_random_state,
    get_all_possible_states,
    magnetization_of_state,
    dict_magnetization_of_all_states,
    hamming_dist,
)

# Example test script - havent tested everything yet

@pytest.mark.parametrize(
    "num_spins",
    [1, 2, 3, 4, 5],
)
def test_get_random_state(num_spins):
    random_state = get_random_state(num_spins)
    assert isinstance(random_state, str)
    assert len(random_state) == num_spins
    assert all(bit in "01" for bit in random_state)


@pytest.mark.parametrize(
    "num_spins, expected_states",
    [
        (1, ["0", "1"]),
        (2, ["00", "01", "10", "11"]),
        (3, ["000", "001", "010", "011", "100", "101", "110", "111"]),
    ],
)
def test_get_all_possible_states(num_spins, expected_states):
    possible_states = get_all_possible_states(num_spins)
    assert possible_states == expected_states



@pytest.mark.parametrize(
    "states, expected_energies, expected_positions",
    [
        (
            [
                MCMCState(bitstring="000", accepted=True, energy=1.0, position=0),
                MCMCState(bitstring="001", accepted=False, energy=2.0, position=1),
                MCMCState(bitstring="010", accepted=True, energy=3.0, position=2),
            ],
            [1.0, 3.0],
            [0, 2],
        )
    ],
)
def test_get_accepted_energies(states, expected_energies, expected_positions):
    chain = MCMCChain(states=states)
    accepted_energies, accepted_positions = chain.get_accepted_energies()

    assert list(accepted_energies) == expected_energies
    assert list(accepted_positions) == expected_positions


@pytest.mark.parametrize(
    "bitstring, expected_magnetization",
    [("010", -(1/3)), ("111", 1), ("000", -1)],
)
def test_magnetization_of_state(bitstring, expected_magnetization):
    magnetization = magnetization_of_state(bitstring)
    assert magnetization == expected_magnetization


@pytest.mark.parametrize(
    "states, expected_dict",
    [
        (
            ["00", "01", "10", "11"],
            {"00": -1.0, "01": 0.0, "10": 0.0, "11": 1.0},
        )
    ],
)
def test_dict_magnetization_of_all_states(states, expected_dict):
    magnetization_dict = dict_magnetization_of_all_states(states)
    assert magnetization_dict == expected_dict


@pytest.mark.parametrize(
    "str1, str2, expected_distance",
    [("110", "101", 2), ("000", "111", 3), ("101", "101", 0)],
)
def test_hamming_dist(str1, str2, expected_distance):
    distance = hamming_dist(str1, str2)
    assert distance == expected_distance

"""

@pytest.mark.parametrize(
    "energy_s, energy_sprime, temperature, bool",
    [
        (1.0, 0.5, 1.0, True),
        (2.0, 2.5, 0.0001, False),
    ],
)
def test_test_accept(energy_s, energy_sprime, temperature, bool):
    result = test_accept(energy_s, energy_sprime, temperature)
    assert result == bool"""