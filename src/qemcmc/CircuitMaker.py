from typing import Any, Union

from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# import qulacs
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
# from qulacs import Observable, QuantumCircuit, QuantumState
# from qulacs.gate import DenseMatrix, X, Z
# from scipy.linalg import expm

from qemcmc.energy_models import EnergyModel, IsingEnergyFunction


class CircuitMaker:
    """Base class for creating quantum circuits for QeMCMC proposals."""

    def __init__(
        self,
        model: Any,
        gamma: Union[float, tuple],
        time: Union[int, tuple],
        delta_time: float = 0.8,
    ):
        self.model = model
        self.couplings = model.couplings
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.n_qubits = model.n

        # Build the Hamiltonians
        # self.H_problem = self._build_problem_hamiltonian()
        # self.H_driver = self._build_driver_hamiltonian()
        # self.H_total = self._build_total_hamiltonian()

    def _build_problem_hamiltonian(self) -> SparsePauliOp:
        """
        Build the problem Hamiltonian from couplings.
        H_problem = Σ h_i Z_i + Σ J_ij Z_i Z_j + ...
        """
        pauli_list = []

        for coupling in self.couplings:
            coupling = np.array(coupling)
            order = coupling.ndim
            if order == 0:
                pauli_list.append(("I" * self.n_qubits, coupling.item()))
                continue

            non_zero_indices = np.transpose(np.nonzero(coupling))
            for index_tuple in non_zero_indices:
                # Skip if indices are not unique (diagonal elements for order >= 2)
                if len(set(index_tuple)) != len(index_tuple):
                    continue

                coeff = coupling[tuple(index_tuple)]
                pauli_str = ["I"] * self.n_qubits
                for i in index_tuple:
                    pauli_str[i] = "Z"
                pauli_list.append(("".join(pauli_str), coeff))

        if not pauli_list:
            return SparsePauliOp(["I" * self.n_qubits], coeffs=[0.0])

        return SparsePauliOp.from_list(pauli_list).simplify()

    def get_qiskit_hamiltonian(self, sign=-1) -> SparsePauliOp:
        """
        Builds a qiskit.quantum_info.SparsePauliOp (Hamiltonian) from the model's couplings.
        """
        pauli_list = []
        coeff_list = []
        n_qubits = self.model.n
        couplings = self.model.couplings

        for coeffs in couplings:
            order = coeffs.ndim
            spin_sign = (-1) ** order
            if order == 0:
                pauli_list.append("I" * n_qubits)
                coeff_list.append(sign * spin_sign * coeffs.item())
                continue

            non_zero_indices = np.transpose(np.nonzero(coeffs))
            for index_tuple in non_zero_indices:
                coeff = coeffs[tuple(index_tuple)]
                pauli_term = ["I"] * n_qubits
                for i in index_tuple:
                    pauli_term[i] = "Z"
                pauli_list.append("".join(pauli_term))
                coeff_list.append(sign * spin_sign * coeff)

        return SparsePauliOp(pauli_list, coeffs=coeff_list).simplify()

    def evolve_problem(self, sign: float = -1.0, reps: int = 1) -> QuantumCircuit:
        """
        Build a circuit implementing time-evolution under the (problem) Hamiltonian
        derived from the model couplings:

            U(t) = exp(-i H t)

        Uses CircuitMaker attributes:
          - self.n_qubits
          - self.time  (if tuple, uses first element; else uses it directly)

        Parameters
        ----------
        sign : float
            Passed through to get_qiskit_hamiltonian(sign=...).
        reps : int
            Trotter repetitions used by PauliEvolutionGate.

        Returns
        -------
        QuantumCircuit
        """
        t = self.time[0] if isinstance(self.time, tuple) else self.time
        if t is None:
            raise ValueError(
                "self.time is None (or empty tuple). Provide an evolution time."
            )

        H = self.get_qiskit_hamiltonian(sign=sign).simplify()

        qc = QuantumCircuit(self.n_qubits)
        qc.append(
            PauliEvolutionGate(H, time=t, synthesis="trotter", reps=reps),
            list(range(self.n_qubits)),
        )
        return qc


# # This code was done using qulacs and runs much quicker! But is no longer supported and is being kept for reference.
# class CircuitMakerIsing(CircuitMaker):
#     """
#     Class which initialises a circuit for a given problem.

#     This class inherits from CircuitMaker and provides a concrete
#     implementation for the Ising model using qulacs.

#     Taken mostly from https://github.com/pafloxy/quMCMC but restructured
#     """

#     def __init__(
#         self,
#         model: "IsingEnergyFunction",
#         gamma: Union[float, tuple],
#         time: Union[int, tuple],
#     ):
#         """
#         Initialise Circuit_Maker object.
#         Args:
#         model: obj
#             energy model object, normally IsingEnergyFunction from energy_models
#         gamma: float or tuple
#             parameter in QeMCMC circuit
#         time: int or tuple
#             parameter in QeMCMC circuit
#         """
#         # Call the parent class constructor to initialize common attributes
#         super().__init__(model, gamma, time)

#         # To fix sign issues
#         if self.model.cost_function_signs[0] == 1:
#             self.h = -self.model.h
#         else:
#             self.h = self.model.h
#         if self.model.cost_function_signs[1] == 1:
#             self.J = -self.model.J
#         else:
#             self.J = self.model.J

#         self.delta_time = 0.8
#         self.n_spins = model.num_spins
#         self.alpha = model.alpha
#         self.pauli_index_list: list = [1, 1]
#         self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

#         # create trotter circuit that is irrelevant of the input string
#         self.qc_evol_h1 = self.fn_qc_h1()
#         self.qc_evol_h2 = self.fn_qc_h2()

#         self.trotter_ckt = self.trottered_qc_for_transition(
#             self.qc_evol_h1, self.qc_evol_h2, self.num_trotter_steps
#         )

#         # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)

#     def build_circuit(self, s: str) -> QuantumCircuit:
#         """
#         Build a quantum circuit for a given bitstring.
#         Args:
#             s (str): The bitstring for which the quantum circuit is to be built.
#         Returns:
#             QuantumCircuit: The combined quantum circuit for the given bitstring.
#         """
#         # build a circuit for a given bitstring
#         qc_s = self.initialise_qc(s)
#         qc_for_s = self.combine_2_qc(qc_s, self.trotter_ckt)  # i can get rid of this!

#         return qc_for_s

#     def get_state_obtained_binary(self, s: str) -> str:
#         """
#         Get the output bitstring s' given an input bitstring s.
#         This method builds a quantum circuit based on the input bitstring `s`,
#         initializes a quantum state (using GPU if available), updates the quantum
#         state with the circuit, and then samples the resulting state to obtain the
#         output bitstring in binary format.
#         Args:
#             s (str): The input bitstring.
#         Returns:
#             str: The output bitstring in binary format.
#         """
#         # get the output bitstring s' given s

#         qc_for_s = self.build_circuit(s)
#         q_state = QuantumState(qubit_count=self.n_spins)
#         q_state.set_zero_state()
#         qc_for_s.update_quantum_state(q_state)

#         state_obtained = q_state.sampling(sampling_count=1)[0]
#         state_obtained_binary = f"{state_obtained:0{self.n_spins}b}"
#         return state_obtained_binary

#     def initialise_qc(self, s: str) -> QuantumCircuit:
#         """
#         Initializes a quantum circuit based on a given binary string.
#         Args:
#             s (str): A binary string where each character represents the initial state of a qubit.
#                     '1' indicates that the qubit should be in the |1⟩ state, and '0' indicates that
#                     the qubit should be in the |0⟩ state.
#         Returns:
#             QuantumCircuit: A quantum circuit with the specified initial states for each qubit.
#         Raises:
#             AssertionError: If the length of the input string `s` does not match the number of qubits.
#         """

#         qc_in = QuantumCircuit(qubit_count=self.n_spins)
#         len_str_in = len(s)
#         assert len_str_in == qc_in.get_qubit_count(), (
#             "len(s) should be equal to number_of_qubits/spins"
#         )

#         for i in range(0, len(s)):
#             if s[i] == "1":
#                 qc_in.add_X_gate(len_str_in - 1 - i)
#         return qc_in

#     def fn_qc_h1(self) -> QuantumCircuit:
#         """
#         Create a Quantum Circuit for time-evolution under Hamiltonian H1.
#         The Hamiltonian H1 is described as:
#         H1 = -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)]
#         This function constructs a quantum circuit that simulates the time evolution
#         under the Hamiltonian H1 for a given time step `delta_time`.
#         Returns:
#             QuantumCircuit: A quantum circuit representing the time evolution under H1.
#         """

#         a = self.gamma
#         b_list = ((self.gamma - 1) * self.alpha) * np.array(self.h)
#         qc_h1 = QuantumCircuit(self.n_spins)
#         for j in range(0, self.n_spins):
#             Matrix = np.round(
#                 expm(
#                     -1j
#                     * self.delta_time
#                     * (a * X(2).get_matrix() + b_list[j] * Z(2).get_matrix())
#                 ),
#                 decimals=6,
#             )

#             unitary_gate = DenseMatrix(index=self.n_spins - 1 - j, matrix=Matrix)
#             qc_h1.add_gate(unitary_gate)

#         return qc_h1

#     def fn_qc_h2(self) -> QuantumCircuit:
#         """
#         Hamiltonian H2 (described in the paper).
#         This function constructs a quantum circuit that simulates the
#         time evolution of a system under the Hamiltonian H2. The Hamiltonian
#         is represented by the interaction matrix `self.J`, and the evolution
#         parameters are determined by `self.gamma` and `self.delta_time`.
#         Returns:
#         -------
#         QuantumCircuit
#             A quantum circuit representing the time evolution under the Hamiltonian H2.
#         """

#         self.n_spins = np.shape(self.J)[0]
#         qc_for_evol_h2 = QuantumCircuit(self.n_spins)
#         upper_triag_without_diag = np.triu(self.J, k=1)
#         theta_array = (
#             -2 * (1 - self.gamma) * self.alpha * self.delta_time
#         ) * upper_triag_without_diag
#         pauli_z_index = [3, 3]  # ZZ
#         for j in range(0, self.n_spins - 1):
#             for k in range(j + 1, self.n_spins):
#                 target_list = [self.n_spins - 1 - j, self.n_spins - 1 - k]
#                 angle = theta_array[j, k]

#                 qc_for_evol_h2.add_multi_Pauli_rotation_gate(
#                     index_list=target_list, pauli_ids=pauli_z_index, angle=angle
#                 )

#         return qc_for_evol_h2

#     def trottered_qc_for_transition(
#         self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int
#     ) -> QuantumCircuit:
#         """
#         Returns a Trotterized quantum circuit.
#         This method constructs a quantum circuit that approximates the evolution under the combined Hamiltonians
#         H1 and H2 using the Trotter-Suzuki decomposition. The resulting circuit is of the form:
#         (evolution_under_h2 X evolution_under_h1)^(num_trotter_steps-1) (evolution under h1).
#         Args:
#             qc_h1 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H1.
#             qc_h2 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H2.
#             num_trotter_steps (int): The number of Trotter steps to use in the decomposition.
#         Returns:
#             QuantumCircuit: The Trotterized quantum circuit representing the combined evolution.
#         """

#         qc_combine = QuantumCircuit(self.n_spins)
#         for _ in range(0, num_trotter_steps - 1):
#             qc_combine.merge_circuit(qc_h1)
#             qc_combine.merge_circuit(qc_h2)

#         qc_combine.merge_circuit(qc_h1)

#         return qc_combine

#     def combine_2_qc(
#         self, init_qc: QuantumCircuit, trottered_qc: QuantumCircuit
#     ) -> QuantumCircuit:
#         """
#         Function to combine 2 quantum circuits of compatible size.

#         """
#         qc_merge = QuantumCircuit(self.n_spins)
#         qc_merge.merge_circuit(init_qc)
#         qc_merge.merge_circuit(trottered_qc)
#         return qc_merge


class CircuitMakerIsing(CircuitMaker):
    """
    Qiskit implementation of CircuitMakerIsing.
    Inherits from CircuitMaker and provides implementation for Ising model.
    """

    def __init__(
        self,
        model: "IsingEnergyFunction",
        gamma: float | tuple,
        time: int | tuple,
    ):
        super().__init__(model, gamma, time)

        # Fix sign issues (Keep logic exactly as provided)
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
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        # Initialize the simulator once
        self.backend = AerSimulator()

        # Create Trotter circuit components
        self.qc_evol_h1 = self.fn_qc_h1()
        self.qc_evol_h2 = self.fn_qc_h2()

        # Combine them into the main evolution block
        self.trotter_ckt = self.trottered_qc_for_transition(
            self.qc_evol_h1, self.qc_evol_h2, self.num_trotter_steps
        )

    def build_circuit(self, s: str) -> QuantumCircuit:
        """
        Build a quantum circuit for a given bitstring.
        """
        # 1. Initialize state |s>
        qc_s = self.initialise_qc(s)
        qc_for_s = qc_s.compose(self.trotter_ckt)

        return qc_for_s

    def get_state_obtained_binary(self, s: str) -> str:
        """
        Get the output bitstring s' given input s using simulation.
        """
        qc = self.build_circuit(s)
        # Add measurement to all qubits to read out the string
        qc.measure_all()
        # Transpile and run
        # Optimization level 0 preserves your exact gate structure (useful for debugging)
        t_qc = transpile(qc, self.backend, optimization_level=0)
        result = self.backend.run(t_qc, shots=1).result()

        # Get counts (e.g., {'101': 1})
        counts = result.get_counts()
        measured_bitstring = list(counts.keys())[0]

        # Return the single sampled bitstring
        return measured_bitstring

    def get_state_obtained_binary_without_transpiling(self, s: str) -> str:
        """
        Efficient simulation using direct matrix evolution (No Transpilation).
        """
        state = Statevector.from_label("0" * self.n_spins)
        state = Statevector.from_label(s)

        # Convert sub-circuits to Operators (matrices)
        op_h1 = Operator(self.qc_evol_h1)
        op_h2 = Operator(self.qc_evol_h2)

        U_step = op_h2.compose(op_h1)

        # Evolve the state
        for _ in range(self.num_trotter_steps - 1):
            state = state.evolve(U_step)

        state = state.evolve(op_h1)
        counts = state.sample_counts(shots=1)
        measured_bitstring = list(counts.keys())[0]

        return measured_bitstring

    def initialise_qc(self, s: str) -> QuantumCircuit:
        """
        Initializes a quantum circuit based on binary string s.
        Preserves original logic: s[i] maps to qubit (N - 1 - i).
        """
        qc_in = QuantumCircuit(self.n_spins)
        len_str_in = len(s)
        assert len_str_in == self.n_spins, "len(s) must match n_spins"

        for i in range(len(s)):
            if s[i] == "1":
                qc_in.x(len_str_in - 1 - i)
        return qc_in

    def fn_qc_h1(self) -> QuantumCircuit:
        """
        Create Quantum Circuit for time-evolution under H1.
        H1 = sum_j [ gamma*X_j - (1-gamma)*alpha*h_j*Z_j ]
        """
        coeff_x = self.gamma
        coeff_z_list = -1 * (1 - self.gamma) * self.alpha * np.array(self.h)

        sparse_list = []

        for j in range(self.n_spins):
            qubit_idx = self.n_spins - 1 - j

            # Add X term: ("Label", [qubits], coefficient)
            sparse_list.append(("X", [qubit_idx], coeff_x))
            # Add Z term
            sparse_list.append(("Z", [qubit_idx], coeff_z_list[j]))

        # Create the full Hamiltonian H1 in one step
        H1 = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=self.n_spins)

        evo_gate = PauliEvolutionGate(H1, time=self.delta_time)

        qc_h1 = QuantumCircuit(self.n_spins)
        qc_h1.append(evo_gate, range(self.n_spins))

        return qc_h1.decompose()

    def fn_qc_h2(self) -> QuantumCircuit:
        """
        Hamiltonian H2 (Interaction terms).
        H2 terms ~ Z_j Z_k
        """
        coeff_prefactor = -(1 - self.gamma) * self.alpha
        sparse_list = []

        # Calculate the coefficient base for the interaction
        # Note: In the original code, 'angle' for rotation was calculated.
        # Here we need the Hamiltonian coefficient C such that U = e^{-i t C ZZ}
        # Original angle theta = -2 * (1-gamma) * alpha * delta_time * J
        # Since R_zz(theta) = e^{-i (theta/2) ZZ},
        # C = -(1-gamma) * alpha * J

        # Iterate strictly upper triangular
        upper_triag_without_diag = np.triu(self.J, k=1)
        rows, cols = np.nonzero(upper_triag_without_diag)

        for j, k in zip(rows, cols):
            q_j = self.n_spins - 1 - j
            q_k = self.n_spins - 1 - k

            J_val = upper_triag_without_diag[j, k]
            coupling_coeff = coeff_prefactor * J_val

            # Add ZZ term: ("ZZ", [qubit_1, qubit_2], coefficient)
            sparse_list.append(("ZZ", [q_j, q_k], coupling_coeff))

        if not sparse_list:
            return QuantumCircuit(self.n_spins)

        H2 = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=self.n_spins)

        evo_gate = PauliEvolutionGate(H2, time=self.delta_time)

        qc_h2 = QuantumCircuit(self.n_spins)
        qc_h2.append(evo_gate, range(self.n_spins))

        return qc_h2.decompose()

    def trottered_qc_for_transition(
        self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int
    ) -> QuantumCircuit:
        """
        Returns a Trotterized quantum circuit.
        Pattern: (H2 H1)^(steps-1) (H1)
        """
        qc_combine = QuantumCircuit(self.n_spins)

        for _ in range(num_trotter_steps - 1):
            qc_combine.compose(qc_h1, inplace=True)
            qc_combine.compose(qc_h2, inplace=True)

        qc_combine.compose(qc_h1, inplace=True)

        return qc_combine


if __name__ == "__main__":
    # 1. Define the physics for your specific problem
    h = np.array([-1.0, -2.0, -3.0])
    J = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -1.5], [0.0, -1.5, 0.0]])
    my_couplings = [h, 0.5 * J]

    # 2. Create an instance of the model
    my_model = EnergyModel(3, couplings=my_couplings)
    cm = CircuitMaker(model=my_model, gamma=0.5, time=2)

    H_new = cm._build_problem_hamiltonian()
    H_qiskit = cm.get_qiskit_hamiltonian()
    H_sparse = my_model.couplings_to_sparse_pauli(3, my_couplings, sign=-1)

    print("=" * 60)
    print("COMPARING get_qiskit_hamiltonian vs couplings_to_sparse_pauli")
    print("=" * 60)
    print(f"\nH_qiskit:\n{H_qiskit}")
    print(f"\nH_sparse:\n{H_sparse}")

    # 4. Test all basis states
    energies_qiskit = []
    energies_sparse = []
    energies_direct = []
    energies_built = []

    print("\n" + "=" * 60)
    print("ENERGIES FOR ALL BASIS STATES")
    print("=" * 60)

    for _state in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        print(f"\nState: {_state}")
        print("-" * 40)

        # Method 1: get_qiskit_hamiltonian expectation value
        sv = Statevector.from_label(_state)  # Reverse for endianness
        energy_qiskit = sv.expectation_value(H_qiskit).real
        print(f"Energy (get_qiskit_hamiltonian): {energy_qiskit:.4f}")
        energies_qiskit.append(energy_qiskit)

        # Method 2: couplings_to_sparse_pauli expectation value
        energy_sparse = sv.expectation_value(H_sparse).real
        print(f"Energy (couplings_to_sparse_pauli): {energy_sparse:.4f}")
        energies_sparse.append(energy_sparse)

        # Method 3: Direct calculation from couplings
        energy_direct = my_model.calculate_energy(_state, my_couplings, sign=-1)
        print(f"Energy (calculate_energy):        {energy_direct:.4f}")
        energies_direct.append(energy_direct)

        energy_built = sv.expectation_value(H_new).real
        print(f"Energy (built Hamiltonian):       {energy_built:.4f}")
        energies_built.append(energy_built)
