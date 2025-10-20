import numpy as np
import pytest
from qemcmc.energy_models import EnergyModel, IsingEnergyFunction


class ConcreteEnergyModel(EnergyModel):
    """Concrete implementation for testing abstract EnergyModel"""
    def calc_an_energy(self, state: str) -> float:
        # Simple energy: count the number of 1s
        return sum(int(c) for c in state)


class TestEnergyModel:
    """Tests for the abstract EnergyModel base class"""
    
    def test_initialization(self):
        """Test basic initialization of EnergyModel"""
        model = ConcreteEnergyModel(n=3, couplings=[], name="Test")
        assert model.n == 3
        assert model.name == "Test"
        assert len(model.couplings) == 0
    
    def test_get_energy(self):
        """Test get_energy method"""
        model = ConcreteEnergyModel(n=3)
        energy = model.get_energy("101")
        assert energy == 2.0
    
    def test_get_energy_type_error(self):
        """Test that get_energy raises TypeError for non-string input"""
        model = ConcreteEnergyModel(n=3)
        with pytest.raises(TypeError, match="State must be a string"):
            model.get_energy([1, 0, 1])
    
    def test_calculate_energy_from_couplings_binary(self):
        """Test energy calculation from couplings with binary representation"""
        h = np.array([1.0, 2.0, 3.0])
        J = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -1.5],
            [0.0, -1.5, 0.0]
        ])
        couplings = [h, J]
        model = ConcreteEnergyModel(n=3, couplings=couplings)
        
        # State '111' -> spins [1, 1, 1]
        energy = model.calculate_energy_from_couplings('111', state_representation='binary', sign=1)
        s = np.array([1, 1, 1])
        field_energy = np.dot(h, s)
        interaction_energy = np.dot(s, J.dot(s))
        expected_energy = field_energy + interaction_energy
        assert np.isclose(energy, expected_energy)

    def test_calculate_energy_from_couplings_spin(self):
        """Test energy calculation from couplings with spin representation"""
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 2.0], [2.0, 0.0]])
        couplings = [h, J]
        model = ConcreteEnergyModel(n=2, couplings=couplings)
        
        spin_state = [1, -1]
        energy = model.calculate_energy_from_couplings(spin_state, state_representation='spin', sign=1)
        s = np.array([1, -1])
        field_energy = np.dot(h, s)
        interaction_energy = np.dot(s, J.dot(s))
        expected_energy = field_energy + interaction_energy
        assert np.isclose(energy, expected_energy)
    
    def test_calculate_energy_from_couplings_negative_sign(self):
        """Test energy calculation with negative sign convention"""
        h = np.array([1.0, 1.0])
        couplings = [h]
        model = ConcreteEnergyModel(n=2, couplings=couplings)
        
        energy_pos = model.calculate_energy_from_couplings('11', state_representation='binary', sign=1)
        energy_neg = model.calculate_energy_from_couplings('11', state_representation='binary', sign=-1)
        
        assert np.isclose(energy_pos, -energy_neg)
    
    def test_calculate_energy_invalid_state_representation(self):
        """Test that invalid state representation raises ValueError"""
        model = ConcreteEnergyModel(n=2, couplings=[np.array([1.0, 1.0])])
        with pytest.raises(ValueError, match="state_representation must be either 'spin' or 'binary'"):
            model.calculate_energy_from_couplings('11', state_representation='invalid')
    
    def test_calculate_energy_spin_type_error(self):
        """Test that spin format requires list input"""
        model = ConcreteEnergyModel(n=2, couplings=[np.array([1.0, 1.0])])
        with pytest.raises(TypeError, match="For 'spin' format, the state must be a list"):
            model.calculate_energy_from_couplings('11', state_representation='spin')
    
    def test_calculate_energy_binary_type_error(self):
        """Test that binary format requires string input"""
        model = ConcreteEnergyModel(n=2, couplings=[np.array([1.0, 1.0])])
        with pytest.raises(TypeError, match="For 'binary' format, the state must be a string"):
            model.calculate_energy_from_couplings([1, -1], state_representation='binary')
    
    def test_calculate_energy_invalid_spin_values(self):
        """Test that spin states must contain only 1 and -1"""
        model = ConcreteEnergyModel(n=2, couplings=[np.array([1.0, 1.0])])
        with pytest.raises(ValueError, match="Spin states in list format must only contain 1 and -1"):
            model.calculate_energy_from_couplings([1, 0], state_representation='spin')
    
    def test_calculate_energy_shape_mismatch(self):
        """Test that coupling tensor shape must match number of spins"""
        h = np.array([1.0, 2.0, 3.0])  # 3 spins
        model = ConcreteEnergyModel(n=2, couplings=[h])  # But n=2
        with pytest.raises(ValueError, match="Shape .* is not compatible"):
            model.calculate_energy_from_couplings('11', state_representation='binary')
    
    def test_calculate_energy_zero_order_coupling(self):
        """Test energy calculation with constant term (0-order coupling)"""
        constant = np.array(5.0)
        h = np.array([1.0, 1.0])
        couplings = [constant, h]
        model = ConcreteEnergyModel(n=2, couplings=couplings)
        
        energy = model.calculate_energy_from_couplings('11', state_representation='binary', sign=1)
        expected = 5.0 + 1.0 + 1.0  # constant + h[0] + h[1]
        assert np.isclose(energy, expected)
    
    def test_get_all_energies(self):
        """Test calculation of all possible state energies"""
        model = ConcreteEnergyModel(n=2)
        energies = model.get_all_energies()
        
        assert len(energies) == 4  # 2^2 states
        assert energies[0] == 0  # '00'
        assert energies[1] == 1  # '01'
        assert energies[2] == 1  # '10'
        assert energies[3] == 2  # '11'
    
    def test_find_lowest_values(self):
        """Test finding lowest unique values and their degeneracies"""
        model = ConcreteEnergyModel(n=2)
        arr = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 2.0])
        
        lowest, degeneracy = model.find_lowest_values(arr, num_values=3)
        
        assert len(lowest) == 3
        assert len(degeneracy) == 3
        assert lowest[0] == 1.0
        assert lowest[1] == 2.0
        assert lowest[2] == 3.0
        assert degeneracy[0] == 2  # 1.0 appears twice
        assert degeneracy[1] == 3  # 2.0 appears three times
        assert degeneracy[2] == 1  # 3.0 appears once
    
    def test_get_lowest_energies(self):
        """Test getting lowest energy states"""
        model = ConcreteEnergyModel(n=2)
        lowest_energies, degeneracies = model.get_lowest_energies(num_states=2)
        
        assert len(lowest_energies) == 2
        assert len(degeneracies) == 2
        assert lowest_energies[0] == 0.0  # '00' has energy 0
        assert degeneracies[0] == 1
        assert lowest_energies[1] == 1.0  # '01' and '10' have energy 1
        assert degeneracies[1] == 2
    
    def test_get_boltzmann_factor(self):
        """Test Boltzmann factor calculation"""
        model = ConcreteEnergyModel(n=2)
        state = "11"
        beta = 1.0
        
        bf = model.get_boltzmann_factor(state, beta)
        expected = np.exp(-1.0 * beta * 2.0)  # energy of '11' is 2
        
        assert np.isclose(bf, expected)
    
    def test_get_boltzmann_factor_from_energy(self):
        """Test Boltzmann factor calculation from energy directly"""
        model = ConcreteEnergyModel(n=2)
        energy = 2.5
        beta = 0.5
        
        bf = model.get_boltzmann_factor_from_energy(energy, beta)
        expected = np.exp(-0.5 * 2.5)
        
        assert np.isclose(bf, expected)


class TestIsingEnergyFunction:
    """Tests for the IsingEnergyFunction class"""
    
    def test_initialization(self):
        """Test initialization of Ising model"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h, name="Test Ising")
        
        assert model.num_spins == 2
        assert model.name == "Test Ising"
        assert np.array_equal(model.J, J)
        assert np.array_equal(model.h, h)
        assert model.lowest_energy is None
        assert model.S is None
    
    def test_initialization_with_initial_states(self):
        """Test that initial states are generated by default"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h)
        
        assert len(model.initial_state) == 100
        for state in model.initial_state:
            assert len(state) == 2
            assert all(c in '01' for c in state)
    
    def test_initialization_no_initial_states(self):
        """Test initialization without initial states"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h, no_initial_states=True)
        
        assert len(model.initial_state) == 0
    
    def test_calc_an_energy_simple(self):
        """Test energy calculation for a simple Ising model"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.0, 0.0])
        
        model = IsingEnergyFunction(J, h)
        
        # State '00' -> spins [-1, -1]
        # Energy = -0.5 * ((-1)*1*(-1)) = -0.5
        energy_00 = model.calc_an_energy('00')
        
        # State '01' -> spins [-1, 1]
        # Energy = -0.5 * ((-1)*1*1) = 0.5
        energy_01 = model.calc_an_energy('01')
        
        assert energy_00 < energy_01
    
    def test_calc_an_energy_with_field(self):
        """Test energy calculation with external field"""
        J = np.array([[0.0, 0.0], [0.0, 0.0]])
        h = np.array([1.0, -1.0])
        
        model = IsingEnergyFunction(J, h)
        
        # State '00' -> spins [-1, -1]
        # Energy = -(1.0*(-1) + (-1.0)*(-1)) = 0
        energy_00 = model.calc_an_energy('00')
        
        # State '11' -> spins [1, 1]
        # Energy = -(1.0*1 + (-1.0)*1) = 0
        energy_11 = model.calc_an_energy('11')
        
        assert np.isclose(energy_00, 0.0)
        assert np.isclose(energy_11, 0.0)
    
    def test_calc_an_energy_type_error(self):
        """Test that calc_an_energy raises TypeError for non-string input"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        model = IsingEnergyFunction(J, h)
        
        with pytest.raises(TypeError, match="State must be a string"):
            model.calc_an_energy([1, 0])
    
    def test_custom_cost_function_signs(self):
        """Test custom cost function signs"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([1.0, 1.0])
        
        model_default = IsingEnergyFunction(J, h, cost_function_signs=[-1, -1])
        model_custom = IsingEnergyFunction(J, h, cost_function_signs=[1, 1])
        
        state = '11'
        energy_default = model_default.calc_an_energy(state)
        energy_custom = model_custom.calc_an_energy(state)
        
        # With opposite signs, energies should have opposite signs
        assert np.isclose(energy_default, -energy_custom)
    
    def test_alpha_calculation(self):
        """Test that alpha scaling factor is calculated correctly"""
        J = np.array([[0.0, 2.0], [2.0, 0.0]])
        h = np.array([1.0, 1.0])
        
        model = IsingEnergyFunction(J, h)
        
        # alpha = sqrt(n) / sqrt(sum(J_ij^2) + sum(h_i^2))
        # J has one unique interaction: J[0,1] = 2.0 (counted once)
        # sum(J_ij^2) = 4.0, sum(h_i^2) = 2.0
        expected_alpha = np.sqrt(2) / np.sqrt(4.0 + 2.0)
        
        assert np.isclose(model.alpha, expected_alpha)
    
    def test_get_J_property(self):
        """Test the get_J property"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h)
        
        assert np.array_equal(model.get_J, J)
    
    def test_get_h_property(self):
        """Test the get_h property"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h)
        
        assert np.array_equal(model.get_h, h)
    
    def test_ising_vs_generic_energy_calculation(self):
        """Test that IsingEnergyFunction matches generic calculation"""
        h = np.array([-1.0, -2.0, -3.0])
        J = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -1.5],
            [0.0, -1.5, 0.0]
        ])
        
        ising_model = IsingEnergyFunction(J, h)
        
        # Test multiple states
        test_states = ['000', '001', '010', '011', '100', '101', '110', '111']
        
        for state in test_states:
            energy_ising = ising_model.calc_an_energy(state)
            
            # Calculate expected energy manually
            spins = np.array([-1 if c == '0' else 1 for c in state])
            expected = -0.5 * np.dot(spins.transpose(), J.dot(spins)) - np.dot(h.transpose(), spins)
            
            assert np.isclose(energy_ising, expected), f"Mismatch for state {state}"
    
    def test_generic_vs_ising_all_states(self):
        """
        Comprehensive test comparing EnergyModel.calculate_energy_from_couplings
        with IsingEnergyFunction.calc_an_energy for all possible states.
        
        This verifies that the generic coupling-based calculation produces
        identical results to the specialized Ising model calculation.
        """
        # Linear coefficients (h vector)
        h = np.array([-1.0, -2.0, -3.0])
        
        # Quadratic coefficients (J matrix) - symmetric
        J = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -1.5],
            [0.0, -1.5, 0.0]
        ])
        
        # Create the list of coupling tensors for generic model
        # Note: Factor of 0.5 accounts for double counting in J matrix
        my_couplings = [h, 0.5 * J]
        
        # Initialize both models
        generic_model = ConcreteEnergyModel(n=3, couplings=my_couplings)
        ising_model = IsingEnergyFunction(J=J, h=h, name="Test Ising Model")
        
        # Test all possible states
        test_states = ['000', '001', '010', '011', '100', '101', '110', '111']
        
        energies_generic = []
        energies_ising = []
        
        for state in test_states:
            # Calculate energy using generic coupling method
            energy_generic = generic_model.calculate_energy_from_couplings(
                state=state, 
                state_representation='binary', 
                sign=-1
            )
            energies_generic.append(energy_generic)
            
            # Calculate energy using Ising model
            energy_ising = ising_model.calc_an_energy(state)
            energies_ising.append(energy_ising)
            
            # Assert they match for each state
            assert np.isclose(energy_generic, energy_ising), (
                f"Energy mismatch for state {state}: "
                f"generic={energy_generic:.6f}, ising={energy_ising:.6f}"
            )
        
        # Verify lowest energies match
        assert np.isclose(min(energies_generic), min(energies_ising)), (
            f"Lowest energy mismatch: "
            f"generic={min(energies_generic):.6f}, ising={min(energies_ising):.6f}"
        )
        
        # Verify highest energies match
        assert np.isclose(max(energies_generic), max(energies_ising)), (
            f"Highest energy mismatch: "
            f"generic={max(energies_generic):.6f}, ising={max(energies_ising):.6f}"
        )
        
        # Verify all energies as arrays
        assert np.allclose(energies_generic, energies_ising), (
            "Energy arrays don't match between generic and Ising models"
        )

    def test_generic_vs_ising_random_system(self):
        """
        Test generic vs Ising energy calculations on a randomly generated system.
        """
        np.random.seed(42)
        n = 5
        
        # Generate random symmetric J matrix
        J = np.random.randn(n, n)
        J = (J + J.T) / 2  # Make symmetric
        np.fill_diagonal(J, 0)  # Zero diagonal
        
        # Generate random h vector
        h = np.random.randn(n)
        
        # Create models
        couplings = [h, 0.5 * J]
        generic_model = ConcreteEnergyModel(n=n, couplings=couplings)
        ising_model = IsingEnergyFunction(J=J, h=h)
        
        # Test random subset of states (testing all 2^5=32 states)
        test_states = [format(i, f'0{n}b') for i in range(2**n)]
        
        for state in test_states:
            energy_generic = generic_model.calculate_energy_from_couplings(
                state=state,
                state_representation='binary',
                sign=-1
            )
            energy_ising = ising_model.calc_an_energy(state)
            
            assert np.isclose(energy_generic, energy_ising, rtol=1e-10), (
                f"Mismatch for state {state}: "
                f"generic={energy_generic}, ising={energy_ising}"
            )
    
    def test_generic_vs_ising_with_different_signs(self):
        """
        Test that sign conventions work correctly in both models.
        """
        h = np.array([1.0, -1.0])
        J = np.array([[0.0, 2.0], [2.0, 0.0]])
        
        # Generic model with sign=-1 (standard Ising convention)
        couplings = [h, 0.5 * J]
        generic_model = ConcreteEnergyModel(n=2, couplings=couplings)
        
        # Ising model with default signs [-1, -1]
        ising_model_neg = IsingEnergyFunction(J=J, h=h, cost_function_signs=[-1, -1])
        
        # Ising model with positive signs [1, 1]
        ising_model_pos = IsingEnergyFunction(J=J, h=h, cost_function_signs=[1, 1])
        
        test_states = ['00', '01', '10', '11']
        
        for state in test_states:
            # Generic with sign=-1 should match Ising with [-1, -1]
            energy_generic_neg = generic_model.calculate_energy_from_couplings(
                state, state_representation='binary', sign=-1
            )
            energy_ising_neg = ising_model_neg.calc_an_energy(state)
            assert np.isclose(energy_generic_neg, energy_ising_neg)
            
            # Generic with sign=+1 should match Ising with [1, 1]
            energy_generic_pos = generic_model.calculate_energy_from_couplings(
                state, state_representation='binary', sign=1
            )
            energy_ising_pos = ising_model_pos.calc_an_energy(state)
            assert np.isclose(energy_generic_pos, energy_ising_pos)
            
            # They should be negatives of each other
            assert np.isclose(energy_generic_neg, -energy_generic_pos)

    def test_get_lowest_energy_caching(self):
        """Test that get_lowest_energy caches the result"""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        
        model = IsingEnergyFunction(J, h)
        model.lowest_energy = -5.0  # Set cached value
        
        result = model.get_lowest_energy()
        assert result == -5.0
    
    def test_three_spin_system(self):
        """Test a three-spin Ising system"""
        J = np.array([
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 0.0]
        ])
        h = np.array([0.1, -0.2, 0.3])
        
        model = IsingEnergyFunction(J, h)
        
        # Just verify it runs and produces reasonable results
        energy = model.calc_an_energy('101')
        assert isinstance(energy, (int, float))
        
        energies = model.get_all_energies()
        assert len(energies) == 8  # 2^3
    

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_spin_system(self):
        """Test a single spin system"""
        J = np.array([[0.0]])
        h = np.array([1.0])
        
        model = IsingEnergyFunction(J, h)
        
        energy_0 = model.calc_an_energy('0')
        energy_1 = model.calc_an_energy('1')
        
        assert energy_0 != energy_1
    
    def test_large_system(self):
        """Test that a larger system can be initialized and calculated"""
        n = 10
        J = np.random.randn(n, n)
        J = (J + J.T) / 2  # Make symmetric
        np.fill_diagonal(J, 0)
        h = np.random.randn(n)
        
        model = IsingEnergyFunction(J, h)
        
        state = '1010101010'
        energy = model.calc_an_energy(state)
        
        assert isinstance(energy, (int, float))
    
    def test_zero_couplings(self):
        """Test system with all zero couplings"""
        J = np.zeros((3, 3))
        h = np.zeros(3)
        
        model = IsingEnergyFunction(J, h)
        
        # All states should have zero energy
        for state in ['000', '001', '010', '111']:
            energy = model.calc_an_energy(state)
            assert np.isclose(energy, 0.0)
    
    def test_high_order_coupling(self):
        """Test higher-order coupling tensors"""
        # 3-body interaction
        h = np.array([1.0, 1.0, 1.0])
        J = np.zeros((3, 3))
        K = np.random.randn(3, 3, 3)  # 3-body coupling
        
        couplings = [h, J, K]
        model = ConcreteEnergyModel(n=3, couplings=couplings)
        
        # Just verify it can calculate without error
        energy = model.calculate_energy_from_couplings('111', state_representation='binary')
        assert isinstance(energy, (int, float, np.floating))
    
    def test_empty_couplings(self):
        """Test model with no couplings"""
        model = ConcreteEnergyModel(n=2, couplings=[])
        
        energy = model.calculate_energy_from_couplings('11', state_representation='binary')
        assert np.isclose(energy, 0.0)