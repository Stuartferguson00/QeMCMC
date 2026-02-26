# Internal
from qemcmc.model import EnergyModel

# External
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

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def get_problem_hamiltonian_OLD(self, couplings, sign=-1):
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

    def get_problem_hamiltonian(self, couplings, sign=-1):
        """
        Construct Problem Hamiltonian from symmetric coupling tensors.
        """
        coeffs = []
        obs = []

        for coupling_tensor in couplings:
            coupling_tensor = np.asarray(coupling_tensor)
            order = coupling_tensor.ndim
            if order == 0:
                continue
            spin_sign = (-1) ** order
            non_zero_indices = np.transpose(np.nonzero(coupling_tensor))
            for index_tuple in non_zero_indices:
                index_tuple = tuple(int(i) for i in index_tuple)

                # skip repeated indices
                if len(set(index_tuple)) != len(index_tuple):
                    continue
                # keep only strictly increasing tuples i1 < i2 < ... < ik
                if index_tuple != tuple(sorted(index_tuple)):
                    continue
                coeff = coupling_tensor[index_tuple]
                if coeff == 0:
                    continue

                term = qml.PauliZ(index_tuple[0])
                for q in index_tuple[1:]:
                    term = term @ qml.PauliZ(q)
                coeffs.append(sign * spin_sign * float(coeff))
                obs.append(term)

        return qml.Hamiltonian(coeffs, obs)

    def get_mixer_hamiltonian(self):
        """Constructs the Mixer Hamiltonian: Σ X_i."""
        return qml.Hamiltonian([1.0] * self.n_qubits, [qml.PauliX(i) for i in range(self.n_qubits)])

    def get_state_vector(self, s: str) -> str:
        """Return the state vector."""
        # Coefficients
        alpha = self.model.calculate_alpha(couplings=self.local_couplings)
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

        state_vector = quantum_evolution(s)
        return state_vector

    def get_sample_from_state_vector(self, s: str) -> str:
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

    def get_sample(self, s: str):
        """Returns a measured sample after time evolution"""
        # Coefficients
        alpha = self.model.calculate_alpha(couplings=self.local_couplings)
        coeff_mixer = self.gamma
        coeff_problem = -(1 - self.gamma) * alpha

        H_total = qml.Hamiltonian([coeff_mixer] + [1.0], [self.get_mixer_hamiltonian(), self.get_problem_hamiltonian(couplings=self.local_couplings, sign=coeff_problem)])

        @qml.qnode(self.dev, shots=1)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, self.time, self.num_trotter_steps)
            return qml.sample()

        # Get the first shot from the sample
        sample = quantum_evolution(s)[0]  # pennylane update doesnt squeeze singletons anymore
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring

    def update(self, s, subgroup_choice, local_couplings, gamma, time) -> str:
        """
        Performs time evolution on coarse grained hamiltonian update to get s' from s
        """
        self._assert_bitstring(s)
        self.gamma = gamma
        self.time = time
        self.local_couplings = local_couplings

        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_sample(s_cg)

        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            s_list[global_index] = s_cg_prime[i]

        return "".join(s_list)

    def _assert_bitstring(self, s, *, length=None):
        # Accept numpy strings etc.
        if not isinstance(s, str):
            raise TypeError(f"bitstring must be of type str, got {type(s)}: {s!r}")

        if length is not None and len(s) != length:
            raise ValueError(f"bitstring must have length {length}, got {len(s)}: {s!r}")

        bad = set(s) - {"0", "1"}
        if bad:
            raise ValueError(f"bitstring must contain only '0'/'1'. Bad chars: {bad}. Value: {s!r}")

        return s
