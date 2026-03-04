import pennylane as qml
import numpy as np


class CircuitMaker:
    """
    Constructs and simulates quantum circuits used to generate QeMCMC proposals.

    This class builds the Hamiltonian corresponding to a given energy model and
    simulates its time evolution using PennyLane. Starting from a classical
    bitstring configuration, the circuit performs Trotterised quantum evolution
    and samples a new configuration from the resulting quantum state.

    The generated sample serves as the proposal state in the quantum-enhanced
    MCMC algorithm.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the problem Hamiltonian.
    gamma : float
        Strength of the transverse-field (mixer) term in the Hamiltonian.
    time : int or tuple[int, int]
        Total evolution time or range of evolution times used in the simulation.
    delta_time : float, optional
        Duration of each Trotter step used in the approximate time evolution.
        Default is 0.8.

    Notes
    -----
    The total Hamiltonian simulated by the circuit is

        H = γ H_mixer + (1 - γ) α H_problem

    where ``H_problem`` encodes the classical energy model and ``H_mixer``
    corresponds to a transverse-field term. The evolution is approximated
    using Trotterisation via ``qml.ApproxTimeEvolution``.
    """

    def __init__(self, model, gamma, time, delta_time=0.8):
        self.model = model
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        self.n_qubits = model.n

        # Calculate number of Trotter steps
        t_val = self.time[0] if isinstance(self.time, tuple) else self.time
        self.num_trotter_steps = int(np.floor((t_val / self.delta_time)))

        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)

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
        alpha = self.model.calculate_alpha(n=self.spin_length, couplings=self.local_couplings)
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
        self.spin_length = len(subgroup_choice)

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
