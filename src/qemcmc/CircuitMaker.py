import numpy as np
import qulacs
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import DenseMatrix,  X, Z
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from typing import Union, Any, List

from qemcmc.energy_models import IsingEnergyFunction, EnergyModel
from qulacsvis import circuit_drawer



class CircuitMaker():
    """Base class for creating quantum circuits for QeMCMC proposals."""

    def __init__(self, model: Any, gamma: Union[float, tuple], time: Union[int, tuple], delta_time: float = 0.8):

        self.model = model
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.n_qubits = model.n

        # Automatically build the full, trottered evolution circuit upon initialization
        self.evolution_circuit = self._create_full_evolution_circuit()

    def build_circuit(self, s: str) -> QuantumCircuit:
        """
        Build the full quantum circuit. 
        It should handle both the initial state preparation and the time evolution.
        This method must be implemented by the subclass. 
        """
        pass

    def _create_full_evolution_circuit(self) -> QuantumCircuit:
        """Creates the complete time-evolution circuit using Trotterization."""
        num_trotter_steps = int(np.floor(self.time / self.delta_time))
        
        # 1. Separate the 1st-order (h) and higher-order couplings
        h_vector = None
        higher_order_couplings = []
        for coeffs in self.model.couplings:
            if coeffs.ndim == 1:
                h_vector = coeffs
            elif coeffs.ndim > 1:
                higher_order_couplings.append(coeffs)
        
        if h_vector is None:
            h_vector = np.zeros(self.n_qubits)

        # 2. Build circuit blocks for one Trotter step
        single_qubit_circuit = self._create_single_qubit_step(h_vector)
        multi_qubit_circuit = self._create_multi_qubit_step(higher_order_couplings)

        # 3. Combine them in a Trotter sequence
        full_circuit = QuantumCircuit(self.n_qubits)
        for _ in range(num_trotter_steps):
            full_circuit.merge_circuit(single_qubit_circuit)
            full_circuit.merge_circuit(multi_qubit_circuit)
            
        return full_circuit

    def _create_single_qubit_step(self, h_vector: np.ndarray) -> QuantumCircuit:
        """
        OPTIMIZATION 2: Combines problem (Z) and mixer (X) terms for each qubit
        into a single, efficient custom gate.
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Coefficients for the X and Z parts of the single-qubit Hamiltonian
        x_coeff = self.gamma
        z_coeffs = (1 - self.gamma) * h_vector
        
        z_gate_matrix = Z(0).get_matrix()
        x_gate_matrix = X(0).get_matrix()

        for i in range(self.n_qubits):
            # Calculate the exact evolution matrix for H = (x_coeff * X + z_coeffs[i] * Z)
            hamiltonian_part = x_coeff * x_gate_matrix + z_coeffs[i] * z_gate_matrix
            evolution_matrix = expm(-1j * self.delta_time * hamiltonian_part)
            
            # Add this custom matrix as a gate for the i-th qubit
            custom_gate = DenseMatrix(i, evolution_matrix)
            qc.add_gate(custom_gate)
            
        return qc

    def _create_multi_qubit_step(self, couplings: List[np.ndarray]) -> QuantumCircuit:
        """
        Automates piece-by-piece circuit construction for higher-order interactions.
        """
        qc = QuantumCircuit(self.n_qubits)
        angle_multiplier = (1 - self.gamma) * self.delta_time
        
        for coeffs in couplings:
            order = coeffs.ndim
            if order < 2: continue # Only handle 2nd-order and higher

            indices_to_iterate = []
            if order == 2:
                # OPTIMIZATION 1: Use upper triangle to avoid double-counting ZZ terms
                # We extract the non-zero elements from the upper triangle only
                upper_triangle = np.triu(coeffs, k=1)
                indices_to_iterate = np.transpose(np.nonzero(upper_triangle))
            else:
                # For order 3 and higher, we iterate through all non-zero elements
                indices_to_iterate = np.transpose(np.nonzero(coeffs))

            for index_tuple in indices_to_iterate:
                coeff = coeffs[tuple(index_tuple)]
                term_angle = 2 * coeff * angle_multiplier
                self._add_pauli_z_string_evolution(qc, term_angle, list(index_tuple))
        return qc

    def _add_pauli_z_string_evolution(self, circuit: QuantumCircuit, angle: float, targets: List[int]):
        """Applies gates for exp(-i*angle/2 * Z_i*Z_j*...*Z_k)."""
        if len(targets) == 1: # Should not happen with new structure, but good practice
            circuit.add_RZ_gate(targets[0], -angle)
            return
        if len(targets) > 1:
            for i in range(len(targets) - 1):
                circuit.add_CNOT_gate(targets[i], targets[-1])
            circuit.add_RZ_gate(targets[-1], -angle)
            for i in range(len(targets) - 2, -1, -1):
                circuit.add_CNOT_gate(targets[i], targets[-1])



    # def _build_hamiltonian(self) -> Observable:
    #     """
    #     Builds a qulacs.Observable (Hamiltonian) from the model's couplings list.
    #     This method converts classical Ising terms into quantum Pauli operators:
    #     - h_i -> h_i * Z_i
    #     - J_ij -> J_ij * Z_i * Z_j
    #     - L_ijk -> L_ijk * Z_i * Z_j * Z_k
    #     and so on.
    #     Returns:
    #         qulacs.Observable: The Hamiltonian operator for the given model.
    #     """
    #     n_qubits = self.model.n
    #     couplings = self.model.couplings
    #     hamiltonian_operator = Observable(n_qubits)

    #     for coeffs in couplings:
    #         order = coeffs.ndim
    #         if order == 0:
    #             hamiltonian_operator.add_operator(coeffs.item(), "I")
    #             continue

    #         non_zero_indices = np.transpose(np.nonzero(coeffs))
    #         for index_tuple in non_zero_indices:
    #             # --- FIX: Convert the index array to a tuple ---
    #             coeff_value = coeffs[tuple(index_tuple)]
    #             pauli_string = " ".join([f"Z {i}" for i in index_tuple])
    #             hamiltonian_operator.add_operator(coeff_value, pauli_string)
                
    #     return hamiltonian_operator

    def get_qiskit_hamiltonian(self) -> SparsePauliOp:
        """
        Builds a qiskit.quantum_info.SparsePauliOp (Hamiltonian) from the model's couplings.
        """
        pauli_list = []
        coeff_list = []
        n_qubits = self.model.n
        couplings = self.model.couplings
    
        for coeffs in couplings:
            order = coeffs.ndim
            if order == 0:
                pauli_list.append("I" * n_qubits)
                coeff_list.append(coeffs.item())
                continue
            
            non_zero_indices = np.transpose(np.nonzero(coeffs))
            for index_tuple in non_zero_indices:
                # --- FIX: Convert the index array to a tuple ---
                coeff = coeffs[tuple(index_tuple)]
                pauli_term = ['I'] * n_qubits
                for i in index_tuple:
                    pauli_term[n_qubits - 1 - i] = 'Z'
                pauli_list.append("".join(pauli_term))
                coeff_list.append(coeff)
                
        return SparsePauliOp(pauli_list, coeffs=coeff_list)


    def get_ground_state_energy(self, hamiltonian: Union[Observable, SparsePauliOp]) -> float:
        """
        Calculates the ground state energy of a given Hamiltonian.
        """
        # if isinstance(hamiltonian, Observable):
        #     # --- The Correct Qulacs Method ---
        #     # 1. Create a Hamiltonian object from the Observable
        #     n_qubits = hamiltonian.get_qubit_count()
        #     hamiltonian_obj = Hamiltonian(n_qubits, hamiltonian)
            
        #     # 2. Get the 0-th (lowest) eigenvalue
        #     ground_state_energy = hamiltonian_obj.get_eigenvalue(0)
        #     return ground_state_energy

        if isinstance(hamiltonian, SparsePauliOp):
            # For Qiskit Hamiltonian
            hamiltonian_matrix = hamiltonian.to_matrix()
            eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix)
            return np.min(eigenvalues)
        
        else:
            raise TypeError("Hamiltonian must be a qulacs.Observable or qiskit.SparsePauliOp")


# Modified CircuitMaker class inheriting from the base class
class CircuitMakerIsing(CircuitMaker):
    """
    Class which initialises a circuit for a given problem.

    This class inherits from CircuitMaker and provides a concrete
    implementation for the Ising model using qulacs.

    Taken mostly from https://github.com/pafloxy/quMCMC but restructured
    """

    def __init__(self, model: 'IsingEnergyFunction', gamma: Union[float, tuple], time: Union[int, tuple], noise_model_dict: Union[dict, None] = None):
        """
        Initialise Circuit_Maker object.
        Args:
        model: obj
            energy model object, normally IsingEnergyFunction from energy_models
        gamma: float or tuple
            parameter in QeMCMC circuit
        time: int or tuple
            parameter in QeMCMC circuit
        """
        # Call the parent class constructor to initialize common attributes
        super().__init__(model, gamma, time)

        self.noise_model_dict = noise_model_dict
        if self.noise_model_dict is not None:
            self.noise_model = self.noise_model_dict.get("noise_model", "depolarising")
            self.noise_prob_one_qubit = self.noise_model_dict.get("noise_prob_one_qubit", 0)
            self.noise_prob_two_qubit = self.noise_model_dict.get("noise_prob_two_qubit", 0)
        else:
            self.noise_model = None
        if self.noise_model != "depolarising" and self.noise_model is not None:
            raise ValueError("Only depolarising (or None) noise model is supported for now")

        # To fix sign issues
        if self.model.cost_function_signs[0] == 1:
            self.h = -self.model.h
        else:
            self.h = self.model.h
        if self.model.cost_function_signs[1] == 1:
            self.J = -self.model.J
        else:
            self.J = self.model.J

        self.delta_time = 0.8
        self.n_spins = model.num_spins
        self.alpha = model.alpha
        self.pauli_index_list:list=[1,1]
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        # create trotter circuit that is irrelevant of the input string
        self.qc_evol_h1 = self.fn_qc_h1()
        self.qc_evol_h2 = self.fn_qc_h2()
        self.trotter_ckt = self.trottered_qc_for_transition(self.qc_evol_h1, self.qc_evol_h2, self.num_trotter_steps)

        # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)

    def build_circuit(self, s:str) -> QuantumCircuit:
        """
        Build a quantum circuit for a given bitstring.
        Args:
            s (str): The bitstring for which the quantum circuit is to be built.
        Returns:
            QuantumCircuit: The combined quantum circuit for the given bitstring.
        """
        #build a circuit for a given bitstring
        qc_s = self.initialise_qc(s)
        qc_for_s = self.combine_2_qc(qc_s, self.trotter_ckt)# i can get rid of this!

        return qc_for_s

    def get_state_obtained_binary(self, s: str) -> str:
        """
        Get the output bitstring s' given an input bitstring s.
        This method builds a quantum circuit based on the input bitstring `s`, 
        initializes a quantum state (using GPU if available), updates the quantum 
        state with the circuit, and then samples the resulting state to obtain the 
        output bitstring in binary format.
        Args:
            s (str): The input bitstring.
        Returns:
            str: The output bitstring in binary format.
        """
        #get the output bitstring s' given s

        qc_for_s = self.build_circuit(s)
        q_state= QuantumState(qubit_count=self.n_spins)
        q_state.set_zero_state()
        qc_for_s.update_quantum_state(q_state)

        state_obtained=q_state.sampling(sampling_count=1)[0]
        state_obtained_binary=f"{state_obtained:0{self.n_spins}b}"
        return state_obtained_binary

    def initialise_qc(self, s : str) -> QuantumCircuit :
        """
        Initializes a quantum circuit based on a given binary string.
        Args:
            s (str): A binary string where each character represents the initial state of a qubit.
                    '1' indicates that the qubit should be in the |1⟩ state, and '0' indicates that
                    the qubit should be in the |0⟩ state.
        Returns:
            QuantumCircuit: A quantum circuit with the specified initial states for each qubit.
        Raises:
            AssertionError: If the length of the input string `s` does not match the number of qubits.
        """

        qc_in=QuantumCircuit(qubit_count=self.n_spins)
        len_str_in = len(s)
        assert len_str_in==qc_in.get_qubit_count(), "len(s) should be equal to number_of_qubits/spins"

        for i in range(0,len(s)):
            if s[i]=="1":
                qc_in.add_X_gate(len_str_in - 1 - i)
        return qc_in

    def fn_qc_h1(self) -> QuantumCircuit :
        """
        Create a Quantum Circuit for time-evolution under Hamiltonian H1.
        The Hamiltonian H1 is described as:
        H1 = -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)]
        This function constructs a quantum circuit that simulates the time evolution
        under the Hamiltonian H1 for a given time step `delta_time`.
        Returns:
            QuantumCircuit: A quantum circuit representing the time evolution under H1.
        """

        a=self.gamma
        b_list = ((self.gamma-1)*self.alpha)* np.array(self.h)
        qc_h1 = QuantumCircuit(self.n_spins)
        for j in range(0, self.n_spins):

            Matrix = np.round(expm(-1j*self.delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)

            unitary_gate=DenseMatrix(index=self.n_spins-1-j,
                            matrix = Matrix)
            
            if self.noise_model == "depolarising" and self.noise_prob_one_qubit > 0:
                qc_h1.add_noise_gate(unitary_gate, "Depolarizing", self.noise_prob_one_qubit)
            else:
                qc_h1.add_gate(unitary_gate)

        return qc_h1

    def fn_qc_h2(self) -> QuantumCircuit :
        
        """
        Hamiltonian H2 (described in the paper).
        This function constructs a quantum circuit that simulates the 
        time evolution of a system under the Hamiltonian H2. The Hamiltonian 
        is represented by the interaction matrix `self.J`, and the evolution 
        parameters are determined by `self.gamma` and `self.delta_time`.
        Returns:
        -------
        QuantumCircuit
            A quantum circuit representing the time evolution under the Hamiltonian H2.
        """

        

        
        self.n_spins=np.shape(self.J)[0]
        qc_for_evol_h2=QuantumCircuit(self.n_spins)
        upper_triag_without_diag=np.triu(self.J,k=1)
        theta_array=(-2*(1-self.gamma)*self.alpha*self.delta_time)*upper_triag_without_diag
        pauli_z_index=[3,3]# ZZ
        for j in range(0,self.n_spins-1):
            for k in range(j+1,self.n_spins):

                target_list=[self.n_spins-1-j,self.n_spins-1-k]
                angle = theta_array[j,k]

                #print(" 2 qubit depolarising isnt working I dont think")
                if self.noise_model == "Depolarising" and self.noise_prob_two_qubit > 0:
                    #print("This bit will not work idk what to do")
                    gate = qulacs.gate.PauliRotation(target_list, pauli_z_index, angle)
                    qc_for_evol_h2.add_noise_gate(gate, "Depolarizing", self.noise_prob_two_qubit)
                else:
                    qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle = angle)
                

        return qc_for_evol_h2

    def trottered_qc_for_transition(self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
        """
        Returns a Trotterized quantum circuit.
        This method constructs a quantum circuit that approximates the evolution under the combined Hamiltonians
        H1 and H2 using the Trotter-Suzuki decomposition. The resulting circuit is of the form:
        (evolution_under_h2 X evolution_under_h1)^(num_trotter_steps-1) (evolution under h1).
        Args:
            qc_h1 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H1.
            qc_h2 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H2.
            num_trotter_steps (int): The number of Trotter steps to use in the decomposition.
        Returns:
            QuantumCircuit: The Trotterized quantum circuit representing the combined evolution.
        """
        
        
        qc_combine=QuantumCircuit(self.n_spins)
        for _ in range(0,num_trotter_steps-1):

            qc_combine.merge_circuit(qc_h1)
            qc_combine.merge_circuit(qc_h2)

        qc_combine.merge_circuit(qc_h1)

        return qc_combine

    def combine_2_qc(self, init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
        """ 
            Function to combine 2 quantum circuits of compatible size.
            
        """
        qc_merge=QuantumCircuit(self.n_spins)
        qc_merge.merge_circuit(init_qc)
        qc_merge.merge_circuit(trottered_qc)
        return qc_merge




if __name__== "__main__":

    # 1. Define the physics for your specific problem
    h = np.array([-1.0, -2.0, -3.0])
    J = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, -1.5],
        [0.0, -1.5, 0.0]
    ])
    my_couplings = [h, 0.5 * J]

    # 2. Create an instance of the model
    my_model = EnergyModel(3, couplings=my_couplings)
    cm = CircuitMaker(model=my_model, gamma=0.5, time=2)


    # 3. Call the new method to get the Qiskit Hamiltonian
    hamiltonian = cm.get_qiskit_hamiltonian()

    print("Hamiltonian built successfully!")
    print(hamiltonian)

    # Get ground state energy using qulacs Hamiltonian
    ground_energy = cm.get_ground_state_energy(hamiltonian)

    print(f"Ground state energy: {ground_energy}")

    ising_model = IsingEnergyFunction(J=J, h=h, name="Test Ising Model")
    cm_ising = CircuitMakerIsing(model=ising_model, gamma=0.5, time=2)
    circuit_ising = cm_ising.build_circuit("101")
    print("Ising Circuit built successfully!")
    print(circuit_ising)
    circuit_drawer(circuit_ising)

    print("Base Circuit built successfully!")
    print(cm.evolution_circuit)
    circuit_drawer(cm.evolution_circuit)


    def get_qulacs_circuit_matrix(circuit: QuantumCircuit) -> np.ndarray:
        """
        Manually constructs the unitary matrix of a Qulacs circuit.

        This is done by applying the circuit to each computational basis state
        and using the resulting state vectors as the columns of the matrix.
        """
        n_qubits = circuit.get_qubit_count()
        dim = 2**n_qubits
        unitary_matrix = np.zeros((dim, dim), dtype=complex)

        # Loop through each computational basis state |i>
        for i in range(dim):
            # Prepare a state in the |i> basis
            state = QuantumState(n_qubits)
            state.set_computational_basis(i)

            # Apply the circuit to the state
            circuit.update_quantum_state(state)

            # The resulting state vector is the i-th column of the unitary
            unitary_matrix[:, i] = state.get_vector()
            
        return unitary_matrix
    

    def are_circuits_equivalent(circuit1: QuantumCircuit, circuit2: QuantumCircuit, tolerance: float = 1e-6) -> bool:
        """
        Checks if two Qulacs circuits are functionally equivalent by comparing their unitary matrices.
        """
        if circuit1.get_qubit_count() != circuit2.get_qubit_count():
            return False

        # Get the unitary matrix for each circuit using the foolproof method
        matrix1 = get_qulacs_circuit_matrix(circuit1)
        matrix2 = get_qulacs_circuit_matrix(circuit2)

        # Compare the matrices element-wise
        return np.allclose(matrix1, matrix2, atol=tolerance)


    print(are_circuits_equivalent(circuit_ising, cm.evolution_circuit))  # Should return True