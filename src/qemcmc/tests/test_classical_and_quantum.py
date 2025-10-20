import unittest
import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp

# Import your EnergyModel and CircuitMaker classes
from qemcmc.energy_models import EnergyModel
from qemcmc.CircuitMaker import CircuitMaker

class TestEnergyEquivalence(unittest.TestCase):
    
    def setUp(self):
        # ... (setup code remains the same) ...
        self.n_qubits = 3
        h = np.array([-1.0, -2.0, -3.0])
        J_symmetric = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -1.5],
            [0.0, -1.5, 0.0]
        ])
        self.couplings = [h, 0.5 * J_symmetric]
        self.sign = -1
        self.model = EnergyModel(n=self.n_qubits, couplings=self.couplings)
        cm = CircuitMaker(model=self.model, gamma=0.5, time=1.0)
        hamiltonian_unsigned = cm.get_qiskit_hamiltonian()
        self.hamiltonian = self.sign * hamiltonian_unsigned

    def test_classical_vs_quantum_energy_for_all_states(self):
        """
        Tests energy equivalence with the 0 -> -1, 1 -> +1 convention.
        """
        print("\n--- Testing Energy Equivalence (0 -> -1, 1 -> +1 convention) ---")
        
        for i in range(2**self.n_qubits):
            bit_string = f"{i:0{self.n_qubits}b}"
            
            with self.subTest(state=bit_string):
                # --- Method 1: Classical Energy Calculation ---
                classical_energy = self.model.calculate_energy_from_couplings(
                    state=bit_string,
                    state_representation='binary',
                    sign=self.sign
                )

                # --- Method 2: Quantum Expectation Value ---
                # To match the classical convention, we must flip the bits for the quantum state.
                # '0' (classical spin -1) must correspond to state |1> (quantum spin -1).
                # '1' (classical spin +1) must correspond to state |0> (quantum spin +1).
                flipped_bit_string = bit_string.translate(str.maketrans('01', '10'))
                quantum_state = Statevector.from_label(flipped_bit_string)
                
                quantum_exp_val = quantum_state.expectation_value(self.hamiltonian).real

                # --- Compare and Assert ---
                print(f"State '{bit_string}': Classical E = {classical_energy:7.4f} | Quantum <H> = {quantum_exp_val:7.4f}")
                self.assertAlmostEqual(classical_energy, quantum_exp_val, places=6)

if __name__ == '__main__':
    unittest.main()