import pennylane as qml
import numpy as np
from qemcmc.model.energy_model import EnergyModel
from typing import List


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

    def __init__(self, model: EnergyModel, delta_time: float = 0.8):
        self.model = model
        self.delta_time = delta_time
        self.n_qubits = model.n
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        #self.dev = qml.device("default.tensor", wires=self.n_qubits, method="mps", max_bond_dim=2, contract="auto-mps")

        self.model_type = model.model_type
        self.devices = {}  # cache devices for dynamic subgroup sizes if needed

    def _get_device(self, num_wires: int):
        """Get or create a PennyLane device for the given number of wires."""
        if num_wires not in self.devices:
            self.devices[num_wires] = qml.device("lightning.qubit", wires=num_wires)
            #self.devices[num_wires] = qml.device("default.tensor", wires=num_wires, method="mps", max_bond_dim=2, contract="auto-mps")
        return self.devices[num_wires]

    def get_problem_hamiltonian(self, couplings: List[np.ndarray], sign: int = 1) -> qml.Hamiltonian:
        """
        Construct the problem Hamiltonian from symmetric coupling tensors.

        This method supports both 'ising' (-1/+1) and 'binary' (0/1) models.

        Parameters
        ----------
        couplings : list of np.ndarray
            A list of coupling tensors.
        sign : int, optional
            A sign to apply to the Hamiltonian. Default is 1.

        Returns
        -------
        qml.Hamiltonian
            The problem Hamiltonian.
        """
        total_hamiltonian = 0.0 * qml.Identity(0)
        for coupling_tensor in couplings:
            coupling_tensor = np.asarray(coupling_tensor)
            order = coupling_tensor.ndim
            if order == 0:
                continue

            spin_sign = (-1) ** order if self.model_type == "ising" else 1
            non_zero_indices = np.transpose(np.nonzero(coupling_tensor))
            for index_tuple in non_zero_indices:
                index_tuple = tuple(int(i) for i in index_tuple)

                if len(set(index_tuple)) != len(index_tuple):  # skip repeated indices
                    continue
                if index_tuple != tuple(sorted(index_tuple)):  # keep only strictly increasing i1 < i2 < ... < ik
                    continue

                coeff = float(coupling_tensor[index_tuple])
                if coeff == 0.0:
                    continue

                if self.model_type == "ising":
                    term = qml.PauliZ(index_tuple[0])
                    for q in index_tuple[1:]:
                        term = term @ qml.PauliZ(q)
                    total_hamiltonian += (sign * spin_sign * coeff) * term

                elif self.model_type == "binary":
                    # 0.5 * (I - Z) for first variable
                    term = 0.5 * (qml.Identity(index_tuple[0]) - qml.PauliZ(index_tuple[0]))
                    # multiply by 0.5 * (I - Z) for rest
                    for q in index_tuple[1:]:
                        next_var = 0.5 * (qml.Identity(q) - qml.PauliZ(q))
                        term = term @ next_var

                    total_hamiltonian += (sign * coeff) * term
        simplified_H = qml.simplify(total_hamiltonian)

        coeffs, ops = simplified_H.terms()
        return qml.Hamiltonian(coeffs, ops)

    def get_mixer_hamiltonian(self, num_wires: int = None) -> qml.Hamiltonian:
        """
        Constructs the Mixer Hamiltonian: Σ X_i.

        This can be for the full system or a subgroup.

        Parameters
        ----------
        num_wires : int, optional
            The number of wires (qubits) for the mixer. If None, uses the total number of qubits.

        Returns
        -------
        qml.Hamiltonian
            The mixer Hamiltonian.
        """
        if num_wires is None:
            num_wires = self.n_qubits
        return qml.Hamiltonian([1.0] * num_wires, [qml.PauliX(i) for i in range(num_wires)])

    def get_state_vector(self, s: str, weights: List[float], time: float, mix_weight: float) -> np.ndarray:
        """
        Evolve the initial state and return the final state vector.

        Parameters
        ----------
        s : str
            Input bitstring representing the initial state.
        weights : list of float
            Coefficients for the problem Hamiltonian terms.
        time : float
            Total evolution time.
        mix_weight : float
            Coefficient for the mixer Hamiltonian (between 0 and 1).

        Returns
        -------
        np.ndarray
            The final state vector.


        Notes
        -----
        The total Hamiltonian simulated by the circuit is a weighted sum of the problem Hamiltonian terms and the mixer Hamiltonian:

        In Ferguson et al. (2025) [arXiv:2506.19538], we use gammas to weight the entire problem Hamiltonian vs the mixer vs the constraint Hamiltonian, such that the total Hamiltonian is:

        H = g_p * H_p + g_m * H_m + g_c * H_c

        but here we allow for separate weights for each coupling tensor term, as well as a separate gamma for the mixer. The total Hamiltonian is then:

        H = (w_b1*H_b1+ w_b2*H_b2 +...+w_b2*H_bm) + g_m * H_m

        In other words, the constraint hamiltonian is absorbed in the coupling list, and weighted by the corresponding gamma in the gammas list.
        This allows for more flexible weighting of different terms, and also allows us to use the same code for both constrained and unconstrained problems (by simply including or excluding the constraint Hamiltonian in the coupling list and adjusting the gammas accordingly).

        Note that it is assumed that each term is already normalised appropriately, so the gammas can be interpreted as the relative weights of each term in the total Hamiltonian.

        """

        num_wires = len(s)
        dev = self._get_device(num_wires)

        if mix_weight < 0 or mix_weight > 1:
            raise ValueError(f"mix_weight must be between 0 and 1. Got {mix_weight}")

        if np.any(np.array(weights) < 0):
            raise ValueError(f"Weights must be non-negative. Got {weights}")

        self.time = time
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        coeff_mixer = mix_weight
        coeff_problem = weights

        H_total = qml.Hamiltonian(
            [coeff_mixer] + list(np.ones(len(self.model.normalised_couplings))),
            [self.get_mixer_hamiltonian(num_wires)]
            + [self.get_problem_hamiltonian(couplings=[self.model.normalised_couplings[i]], sign=coeff_problem[i]) for i in range(len(self.model.normalised_couplings))],
        )

        @qml.qnode(dev)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, self.time, self.num_trotter_steps)
            return qml.state()

        state_vector = quantum_evolution(s)
        return state_vector

    def get_sample(self, s_cg: str, time: float, mix_weight: float, local_couplings: list, weights: List[float] = None) -> str:
        """
        Generate a single sample by evolving the system and measuring.

        Parameters
        ----------
        s_cg : str
            Input bitstring for the subgroup.
        time : float
            Total evolution time.
        mix_weight : float
            Coefficient for the mixer Hamiltonian.
        local_couplings : list
            Coupling tensors for the subgroup.
        weights : list of float, optional
            Coefficients for the problem Hamiltonian terms. Defaults to ones.

        Returns
        -------
        str
            A single bitstring sample.
        """
        if weights is None:
            weights = np.ones(len(local_couplings))

        num_wires = len(s_cg)
        dev = self._get_device(num_wires)

        if mix_weight < 0 or mix_weight > 1:
            raise ValueError(f"mix_weight must be between 0 and 1. Got {mix_weight}")

        if np.any(np.array(weights) < 0):
            raise ValueError(f"Weights must be non-negative. Got {weights}")

        num_trotter_steps = int(np.floor((time / self.delta_time)))

        coeff_mixer = mix_weight
        coeff_problem = weights

        # coeff_problem = [coeff for i, coeff in enumerate(coeff_problem) if local_couplings[i].ndim > 0 and np.any(local_couplings[i] != 0)]
        # local_couplings = [coupling for coupling in local_couplings if coupling.ndim > 0 and np.any(coupling != 0)]
        if len(local_couplings) == 0:
            # If there are no local couplings, just return the input state
            print("no non-zero local couplings, skipping evolution and returning input state")
            return s_cg

        H_total = qml.Hamiltonian(
            [coeff_mixer] + list(np.ones(len(local_couplings))),
            [self.get_mixer_hamiltonian(num_wires)] + [self.get_problem_hamiltonian(couplings=[local_couplings[i]], sign=coeff_problem[i]) for i in range(len(local_couplings))],
        )

        # set qnode to use our device with dynamically chosen wires
        @qml.qnode(dev, shots=1)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, time, num_trotter_steps)
            return qml.sample()
        """print(f"Simulating quantum circuit with {num_wires} qubits, time={time}, num_trotter_steps={num_trotter_steps}")
        compiled_circuit = qml.compile(quantum_evolution)

        specs_dict = qml.specs(compiled_circuit)(s_cg)
        specs_dict = specs_dict['resources']
        # 3. Print the relevant metrics
        print(f"Circuit Depth: {specs_dict['depth']}")
        print(f"Total Gates:   {specs_dict['num_gates']}")

        # To see the count of two-qubit gates specifically:
        two_qubit_gates = specs_dict['gate_sizes'].get(2, 0)
        print(f"Two-qubit Gates: {two_qubit_gates}")

        # To see the breakdown by gate type (e.g., CNOT, RZ, etc.)
        print("\nGate Breakdown:")
        for gate, count in specs_dict['gate_types'].items():
            print(f"- {gate}: {count}")"""
        # Get the first shot from the sample
        compiled_circuit = qml.compile(quantum_evolution)
        sample = compiled_circuit(s_cg)[0]
        #sample = quantum_evolution(s_cg)[0]
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring

    def update(self, s: str, subgroup_choice: List[int], local_couplings: list, gamma: float, time: float) -> str:
        """
        Update a bitstring by evolving a subgroup.

        This performs a time evolution on a coarse-grained Hamiltonian to get s' from s.

        Parameters
        ----------
        s : str
            The initial bitstring.
        subgroup_choice : list of int
            Indices of the subgroup to evolve.
        local_couplings : list
            Coupling tensors for the subgroup.
        gamma : float
            Mixing parameter.
        time : float
            Evolution time.

        Returns
        -------
        str
            The updated bitstring s'.
        """

        self._assert_bitstring(s)

        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_sample(s_cg, time, gamma, local_couplings)

        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            s_list[global_index] = s_cg_prime[i]

        return "".join(s_list)

    def _assert_bitstring(self, s: str, *, length: int = None):
        """
        Validate that s is a bitstring.

        Parameters
        ----------
        s : str
            The string to validate.
        length : int, optional
            The expected length of the string.

        Raises
        ------
        TypeError
            If s is not a string.
        ValueError
            If s contains characters other than '0' or '1', or if its length is wrong.
        """
        if not isinstance(s, str):
            raise TypeError(f"bitstring must be of type str, got {type(s)}: {s!r}")

        if length is not None and len(s) != length:
            raise ValueError(f"bitstring must have length {length}, got {len(s)}: {s!r}")

        bad = set(s) - {"0", "1"}
        if bad:
            raise ValueError(f"bitstring must contain only '0'/'1'. Bad chars: {bad}. Value: {s!r}")

        return s
