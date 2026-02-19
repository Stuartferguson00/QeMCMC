# Internal package imports
from qemcmc.model import EnergyModel

# External package imports
import pennylane as qml
import numpy as np
from typing import Any, Union
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector


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
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        self.backend = AerSimulator()

    def initialise_qc(self, s: str) -> QuantumCircuit:
        """Standard big-endian initialization: s[i] maps to qubit (N - 1 - i)."""
        qc = QuantumCircuit(self.n_qubits)
        for i, bit in enumerate(s):
            if bit == "1":
                qc.x(self.n_qubits - 1 - i)
        return qc

    def update(self, s: str) -> str:
        """
        Performs time evolution on coarse grained hamiltonian update to get s' from s
        """
        # 0. auto select the gamma and t
        g_step = self.gamma
        if isinstance(self.gamma, tuple):
            g_step = np.random.uniform(min(self.gamma), max(self.gamma))
        t_step = self.time
        if isinstance(self.time, tuple):
            t_step = np.random.randint(min(self.time), max(self.time) + 1)

        # 1. choose a one of the given subgroups based on probabilities
        idx = np.random.choice(len(self.model.subgroups), p=self.model.subgroup_probs)
        subgroup_choice = self.model.subgroups[idx]

        # 2. calculate couplings, and initialize a small model and circuit maker for that subgroup
        local_couplings = self.model.get_subgroup_couplings(subgroup_choice, s)
        local_model = EnergyModel(n=len(subgroup_choice), couplings=local_couplings)
        local_CM = PennyLaneCircuitMaker(local_model, g_step, t_step)

        # 3. get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = local_CM.get_state(s_cg)

        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            s_list[global_index] = s_cg_prime[i]

        return "".join(s_list)

    def get_problem_hamiltonian(self, sign=-1) -> SparsePauliOp:
        """Builds the Problem Hamiltonian (Z-terms) from model's couplings."""
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

    def get_mixer_hamiltonian(self) -> SparsePauliOp:
        """Returns the transverse field mixer: H_mixer = Σ X_i."""
        pauli_list = []
        for i in range(self.n_qubits):
            p_str = ["I"] * self.n_qubits
            p_str[self.n_qubits - 1 - i] = "X"
            pauli_list.append(("".join(p_str), 1.0))
        return SparsePauliOp.from_list(pauli_list)

    def get_state(self, s: str) -> str:
        """
        Get the output bitstring s' given input s using simulation.
        Workflow: Initialize circuit -> Evolve -> Measure.
        """
        qc = self.initialise_qc(s)
        alpha = self.model.alpha
        coeff_mixer = self.gamma
        coeff_problem = -(1 - self.gamma) * alpha

        H_mixer = self.get_mixer_hamiltonian() * coeff_mixer
        H_problem = self.get_problem_hamiltonian(sign=coeff_problem)
        H_total = (H_mixer + H_problem).simplify()

        synthesis = SuzukiTrotter(reps=self.num_trotter_steps)
        evolution_gate = PauliEvolutionGate(H_total, time=self.time, synthesis=synthesis)

        qc.append(evolution_gate, range(self.n_qubits))
        # qc.measure_all()

        # t_qc = transpile(qc, self.backend, optimization_level=0)
        # result = self.backend.run(t_qc, shots=1).result()
        # return list(result.get_counts().keys())[0]
        state_vector = Statevector.from_instruction(qc)

        # Return as a numpy array so np.abs(state)**2 works in your loop
        return np.array(state_vector.data)


class PennyLaneCircuitMaker:
    def __init__(self, model, gamma, time, delta_time=0.8):
        self.model = model
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.n_qubits = model.n

        # Calculate number of Trotter steps
        t_val = self.time[0] if isinstance(self.time, tuple) else self.time
        self.num_trotter_steps = int(np.floor((t_val / self.delta_time)))

        # 'default.qubit' is the standard high-performance CPU simulator
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.dev2 = qml.device("lightning.qubit", wires=self.n_qubits, shots=1)

    def get_problem_hamiltonian(self, couplings, sign=-1):
        """Constructs the Problem Hamiltonian using PennyLane observables."""
        coeffs = []
        obs = []

        for coupling_tensor in couplings:
            order = coupling_tensor.ndim
            spin_sign = (-1) ** order
            non_zero_indices = np.transpose(np.nonzero(coupling_tensor))

            for index_tuple in non_zero_indices:
                coeff = coupling_tensor[tuple(index_tuple)]
                operators = [qml.PauliZ(i) for i in index_tuple]
                if len(operators) > 1:
                    term = operators[0]
                    for op in operators[1:]:
                        term = term @ op
                else:
                    term = operators[0]

                coeffs.append(sign * spin_sign * coeff)
                obs.append(term)

        return qml.Hamiltonian(coeffs, obs)

    def get_mixer_hamiltonian(self):
        """Constructs the Mixer Hamiltonian: Σ X_i."""
        return qml.Hamiltonian([1.0] * self.n_qubits, [qml.PauliX(i) for i in range(self.n_qubits)])

    def get_state_vector(self, s: str) -> str:
        """Return the state vector."""
        # Coefficients
        alpha = getattr(self.model, "alpha", 1.0)
        coeff_mixer = self.gamma
        coeff_problem = -(1 - self.gamma) * alpha

        H_total = qml.Hamiltonian([coeff_mixer] + [1.0], [self.get_mixer_hamiltonian(), self.get_problem_hamiltonian(couplings=self.local_couplings, sign=coeff_problem)])

        @qml.qnode(self.dev)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, self.time, self.num_trotter_steps)
            return qml.state()

        # Run and convert the sampled array [0, 1, 0] back to "010"
        state_vector = quantum_evolution(s)
        return state_vector

    def get_s_prime(self, s: str) -> str:
        """Returns a single sampled bitstring s' using the quantum distribution."""
        # Get the full state vector probabilities
        state_vector = self.get_state_vector(s)  # This returns the complex amplitudes
        probs = np.abs(state_vector) ** 2

        # Sample one index based on the probabilities
        n_states = len(probs)
        idx = np.random.choice(n_states, p=probs)

        # Convert that index back to a bitstring (e.g., 3 -> "011")
        s_prime = np.binary_repr(idx, width=self.model.n)
        return s_prime

    def get_s_prime_alt(self, s: str):
        """Returns a measured sample after time evolution"""
        # Coefficients
        alpha = getattr(self.model, "alpha", 1.0)
        coeff_mixer = self.gamma
        coeff_problem = -(1 - self.gamma) * alpha

        H_total = qml.Hamiltonian([coeff_mixer] + [1.0], [self.get_mixer_hamiltonian(), self.get_problem_hamiltonian(couplings=self.local_couplings, sign=coeff_problem)])

        @qml.qnode(self.dev2)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, self.time, self.num_trotter_steps)
            return qml.sample()

        sample = quantum_evolution(s)
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring

    def update(self, s, subgroup_choice) -> str:
        """
        Performs time evolution on coarse grained hamiltonian update to get s' from s
        """
        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_s_prime(s_cg)
        # s_cg_prime = self.get_s_prime_alt(s_cg)

        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            # Now s_cg_prime[i] is a character ("0" or "1"), not a complex number
            s_list[global_index] = s_cg_prime[i]

        return "".join(s_list)


if __name__ == "__main__":
    # 1. Define the physics for your specific problem
    h = np.array([-1.0, -2.0, -3.0])
    J = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -1.5], [0.0, -1.5, 0.0]])
    my_couplings = [h, 0.5 * J]

    # 2. Create an instance of the model
    my_model = EnergyModel(3, couplings=my_couplings)
    cm = CircuitMaker(model=my_model, gamma=0.5, time=2)
    cm2 = PennyLaneCircuitMaker(model=my_model, gamma=0.5, time=2)

    H_qiskit = cm.get_problem_hamiltonian()
    H_sparse = my_model.couplings_to_sparse_pauli(3, my_couplings, sign=-1)
    H_pennylane = cm2.get_problem_hamiltonian()

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def get_pl_energy(state_str):
        # Initialize the state in PennyLane (Big-Endian)
        for i, bit in enumerate(state_str):
            if bit == "1":
                qml.PauliX(wires=i)
        # Return expectation value of your Hamiltonian
        return qml.expval(H_pennylane)

    print("=" * 60)
    print("COMPARING get_problem_hamiltonian vs couplings_to_sparse_pauli")
    print("=" * 60)
    print(f"\nH_qiskit:\n{H_qiskit}")
    print(f"\nH_sparse:\n{H_sparse}")

    print("\n" + "=" * 60)
    print("ENERGIES FOR ALL BASIS STATES")
    print("=" * 60)

    for _state in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        print(f"\nState: {_state}")
        print("-" * 40)

        # Method 1: get_problem_hamiltonian expectation value
        sv = Statevector.from_label(_state)  # Reverse for endianness
        energy_qiskit = sv.expectation_value(H_qiskit).real
        print(f"Energy (QISKIT): {energy_qiskit:.4f}")

        # Method 2: couplings_to_sparse_pauli expectation value
        energy_sparse = sv.expectation_value(H_sparse).real
        print(f"Energy (couplings_to_sparse_pauli): {energy_sparse:.4f}")

        # Method 3: Direct calculation from couplings
        energy_direct = my_model.calculate_energy(_state, my_couplings, sign=-1)
        print(f"Energy (calculate_energy):        {energy_direct:.4f}")

        # Method 4: Pennylane hamiltonian expectation value
        energy_penny = float(get_pl_energy(_state))
        print(f"Energy PENNYLANE (get_problem_hamiltonian): {energy_penny:.4f}")
