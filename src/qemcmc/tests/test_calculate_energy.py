import unittest

import numpy as np

from qemcmc.energy_models import EnergyModel


class TestEnergyModel(unittest.TestCase):
    """
    Unit tests for EnergyModel.calculate_energy method.
    """

    def setUp(self):
        """Set up test fixtures - common test data."""
        # Simple 3-qubit Ising model
        self.h = np.array([-1.0, -2.0, -3.0])
        self.J = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -1.5], [0.0, -1.5, 0.0]])
        self.couplings_simple = [self.h, 0.5 * self.J]

        # Create model
        self.model = EnergyModel(n=3, couplings=self.couplings_simple)

    def test_string_input_binary(self):
        """Test that string input works correctly with binary spin_type."""
        state_str = "011"
        energy = self.model.calculate_energy(
            state_str, self.couplings_simple, spin_type="binary"
        )
        self.assertIsInstance(energy, (float, np.floating))
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_list_input_binary(self):
        """Test that list input works correctly with binary spin_type."""
        state_list = [0, 1, 1]
        energy = self.model.calculate_energy(
            state_list, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_tuple_input_binary(self):
        """Test that tuple input works correctly with binary spin_type."""
        state_tuple = (0, 1, 1)
        energy = self.model.calculate_energy(
            state_tuple, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_numpy_array_input_binary(self):
        """Test that numpy array input works correctly with binary spin_type."""
        state_array = np.array([0, 1, 1])
        energy = self.model.calculate_energy(
            state_array, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_list_input_spin(self):
        """Test that list input works correctly with spin spin_type."""
        state_list = [-1, 1, 1]
        energy = self.model.calculate_energy(
            state_list, self.couplings_simple, spin_type="spin"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_tuple_input_spin(self):
        """Test that tuple input works correctly with spin spin_type."""
        state_tuple = (-1, 1, 1)
        energy = self.model.calculate_energy(
            state_tuple, self.couplings_simple, spin_type="spin"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_numpy_array_input_spin(self):
        """Test that numpy array input works correctly with spin spin_type."""
        state_array = np.array([-1, 1, 1])
        energy = self.model.calculate_energy(
            state_array, self.couplings_simple, spin_type="spin"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_binary_spin_equivalence(self):
        """Test that binary [0,1,1] and spin [-1,1,1] give the same energy."""
        state_binary = [0, 1, 1]
        state_spin = [-1, 1, 1]

        energy_binary = self.model.calculate_energy(
            state_binary, self.couplings_simple, spin_type="binary"
        )
        energy_spin = self.model.calculate_energy(
            state_spin, self.couplings_simple, spin_type="spin"
        )

        self.assertAlmostEqual(energy_binary, energy_spin, places=10)

    def test_all_binary_states_consistency(self):
        """Test that all input types give the same energy for the same binary state."""
        state_str = "101"
        state_list = [1, 0, 1]
        state_tuple = (1, 0, 1)
        state_array = np.array([1, 0, 1])

        energy_str = self.model.calculate_energy(
            state_str, self.couplings_simple, spin_type="binary"
        )
        energy_list = self.model.calculate_energy(
            state_list, self.couplings_simple, spin_type="binary"
        )
        energy_tuple = self.model.calculate_energy(
            state_tuple, self.couplings_simple, spin_type="binary"
        )
        energy_array = self.model.calculate_energy(
            state_array, self.couplings_simple, spin_type="binary"
        )

        self.assertAlmostEqual(energy_str, energy_list, places=10)
        self.assertAlmostEqual(energy_list, energy_tuple, places=10)
        self.assertAlmostEqual(energy_tuple, energy_array, places=10)

    def test_all_spin_states_consistency(self):
        """Test that all input types give the same energy for the same spin state."""
        state_list = [1, -1, 1]
        state_tuple = (1, -1, 1)
        state_array = np.array([1, -1, 1])

        energy_list = self.model.calculate_energy(
            state_list, self.couplings_simple, spin_type="spin"
        )
        energy_tuple = self.model.calculate_energy(
            state_tuple, self.couplings_simple, spin_type="spin"
        )
        energy_array = self.model.calculate_energy(
            state_array, self.couplings_simple, spin_type="spin"
        )

        self.assertAlmostEqual(energy_list, energy_tuple, places=10)
        self.assertAlmostEqual(energy_tuple, energy_array, places=10)

    def test_all_zeros_binary(self):
        """Test energy calculation for all zeros state in binary."""
        state = "000"
        energy = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="binary"
        )
        # All zeros binary = all -1 spin, so should match all -1 spin state
        self.assertAlmostEqual(energy, 5.0, places=5)

    def test_all_ones_binary(self):
        """Test energy calculation for all ones state in binary."""
        state = "111"
        energy = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -7.0, places=5)

    def test_all_minus_one_spin(self):
        """Test energy calculation for all -1 spin state."""
        state = [-1, -1, -1]
        energy = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="spin"
        )
        self.assertAlmostEqual(energy, 5.0, places=5)

    def test_all_plus_one_spin(self):
        """Test energy calculation for all +1 spin state."""
        state = [1, 1, 1]
        energy = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="spin"
        )
        self.assertAlmostEqual(energy, -7.0, places=5)

    def test_linear_only_binary(self):
        """Test energy with only linear terms (1D coupling) using binary."""
        h_only = np.array([1.0, -2.0, 3.0])
        couplings = [h_only]

        # Binary [1, 0, 1] -> Spin [1, -1, 1]
        # Energy = 1*1 + (-2)*(-1) + 3*1 = 1 + 2 + 3 = 6
        state = [1, 0, 1]
        energy = self.model.calculate_energy(state, couplings, spin_type="binary")
        self.assertAlmostEqual(energy, 6.0, places=5)

    def test_linear_only_spin(self):
        """Test energy with only linear terms (1D coupling) using spin."""
        h_only = np.array([1.0, -2.0, 3.0])
        couplings = [h_only]

        # Spin [1, -1, 1]
        # Energy = 1*1 + (-2)*(-1) + 3*1 = 1 + 2 + 3 = 6
        state = [1, -1, 1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")
        self.assertAlmostEqual(energy, 6.0, places=5)

    def test_quadratic_only(self):
        """Test energy with only quadratic terms (2D coupling)."""
        J_only = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
        couplings = [J_only]

        # Spin [1, -1, 1]
        # Energy = 2*1*(-1) + 2*(-1)*1 + (-1)*(-1)*1 + (-1)*1*(-1)
        #        = -2 + (-2) + 1 + 1 = -2
        state = [1, -1, 1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")
        self.assertAlmostEqual(energy, -2.0, places=5)

    def test_cubic_terms(self):
        """Test energy with cubic terms (3D coupling)."""
        h3 = np.array([1.0, -0.5, 2.0])
        J3 = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.5], [0.0, 0.5, 0.0]])

        # Cubic coupling tensor
        K3 = np.zeros((3, 3, 3))
        K3[0, 1, 2] = 0.3
        K3[1, 0, 2] = 0.3
        K3[2, 0, 1] = 0.3
        K3[0, 2, 1] = 0.3
        K3[1, 2, 0] = 0.3
        K3[2, 1, 0] = 0.3

        couplings_cubic = [h3, 0.5 * J3, K3 / 6.0]

        # Test binary input
        state_binary = "110"
        energy_bin = self.model.calculate_energy(
            state_binary, couplings_cubic, spin_type="binary"
        )

        # Test spin input (same physical state)
        state_spin = [1, 1, -1]
        energy_spin = self.model.calculate_energy(
            state_spin, couplings_cubic, spin_type="spin"
        )

        # They should match
        self.assertAlmostEqual(energy_bin, energy_spin, places=10)

    def test_multiple_linear_couplings(self):
        """Test with multiple 1D arrays in couplings list."""
        h1 = np.array([1.0, -1.0, 2.0])
        h2 = np.array([0.5, 0.5, -0.5])
        couplings = [h1, h2]

        state = [1, 1, -1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")

        # Energy = (1*1 + (-1)*1 + 2*(-1)) + (0.5*1 + 0.5*1 + (-0.5)*(-1))
        #        = (1 - 1 - 2) + (0.5 + 0.5 + 0.5)
        #        = -2 + 1.5 = -0.5
        self.assertAlmostEqual(energy, -0.5, places=5)

    def test_multiple_quadratic_couplings(self):
        """Test with multiple 2D arrays in couplings list."""
        J1 = np.array([[0.0, 1.0], [1.0, 0.0]])
        J2 = np.array([[0.0, -0.5], [-0.5, 0.0]])
        couplings = [J1, J2]

        state = [1, -1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")

        # J1 contribution: 1*1*(-1) + 1*(-1)*1 = -1 + (-1) = -2
        # J2 contribution: (-0.5)*1*(-1) + (-0.5)*(-1)*1 = 0.5 + 0.5 = 1
        # Total: -2 + 1 = -1
        self.assertAlmostEqual(energy, -1.0, places=5)

    def test_empty_couplings(self):
        """Test with empty couplings list."""
        couplings = []
        state = [1, 0, 1]
        energy = self.model.calculate_energy(state, couplings, spin_type="binary")
        self.assertAlmostEqual(energy, 0.0, places=10)

    def test_default_spin_type(self):
        """Test that default spin_type is 'binary'."""
        state = [0, 1, 1]
        energy_default = self.model.calculate_energy(state, self.couplings_simple)
        energy_explicit = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy_default, energy_explicit, places=10)

    def test_large_state(self):
        """Test with a larger state (10 qubits)."""
        n = 10
        h = np.random.randn(n)
        J = np.random.randn(n, n)
        J = (J + J.T) / 2  # Make symmetric
        couplings = [h, 0.5 * J]

        # Binary state
        state_binary = "1010101010"
        energy = self.model.calculate_energy(
            state_binary, couplings, spin_type="binary"
        )
        self.assertIsInstance(energy, (float, np.floating))

    def test_float_values_in_state(self):
        """Test that float values (0.0, 1.0) work correctly."""
        state_float = [0.0, 1.0, 1.0]
        energy = self.model.calculate_energy(
            state_float, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_integer_state_binary(self):
        """Test with integer numpy array."""
        state = np.array([0, 1, 1], dtype=int)
        energy = self.model.calculate_energy(
            state, self.couplings_simple, spin_type="binary"
        )
        self.assertAlmostEqual(energy, -6.0, places=5)

    def test_all_eight_binary_states(self):
        """Test all 8 possible states for 3 qubits and verify energies match between formats."""
        binary_states = ["000", "001", "010", "011", "100", "101", "110", "111"]
        spin_states = [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]

        for binary_state, spin_state in zip(binary_states, spin_states):
            energy_binary = self.model.calculate_energy(
                binary_state, self.couplings_simple, spin_type="binary"
            )
            energy_spin = self.model.calculate_energy(
                spin_state, self.couplings_simple, spin_type="spin"
            )

            self.assertAlmostEqual(
                energy_binary,
                energy_spin,
                places=10,
                msg=f"Energy mismatch for binary {binary_state} vs spin {spin_state}",
            )


class TestEnergyModelQuarticAndHigher(unittest.TestCase):
    """Test cases for quartic (4th order) and higher order terms."""

    def setUp(self):
        """Set up test fixtures for higher-order tests."""
        self.model = EnergyModel(n=4)

    def test_quartic_terms(self):
        """Test energy with quartic (4th order) terms."""
        # 4-variable system
        h = np.array([0.5, -0.5, 1.0, -1.0])

        # Quartic tensor (4D)
        Q = np.zeros((4, 4, 4, 4))
        Q[0, 1, 2, 3] = 0.1
        # Add all permutations for symmetry
        from itertools import permutations

        for perm in permutations([0, 1, 2, 3]):
            Q[perm] = 0.1

        couplings = [h, Q / 24.0]  # Normalize by number of permutations

        state = [1, 1, -1, 1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")
        self.assertIsInstance(energy, (float, np.floating))

    def test_fifth_order_terms(self):
        """Test energy with 5th order terms."""
        n = 5
        h = np.array([1.0, -1.0, 0.5, -0.5, 2.0])

        # 5th order tensor
        T5 = np.zeros((n, n, n, n, n))
        T5[0, 1, 2, 3, 4] = 0.05

        couplings = [h, T5]

        state = [1, -1, 1, 1, -1]
        energy = self.model.calculate_energy(state, couplings, spin_type="spin")
        self.assertIsInstance(energy, (float, np.floating))
