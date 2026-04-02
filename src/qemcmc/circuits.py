import pennylane as qml
import numpy as np
from qemcmc.model.energy_model import EnergyModel

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

    def __init__(self, model:EnergyModel, delta_time=0.8):
        self.model = model
        self.delta_time = delta_time
        self.n_qubits = model.n


        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.model_type = model.model_type
        # cache devices for dynamic subgroup sizes if needed
        self.devices = {}

    def _get_device(self, num_wires: int):
        """Get or create a PennyLane device for the given number of wires."""
        if num_wires not in self.devices:
            self.devices[num_wires] = qml.device("lightning.qubit", wires=num_wires)
        return self.devices[num_wires]

    # def get_problem_hamiltonian(self, couplings, sign=-1):
    #     """
    #     Construct Problem Hamiltonian from symmetric coupling tensors.
    #     """
    #     coeffs = []
    #     obs = []

    #     for coupling_tensor in couplings:
    #         coupling_tensor = np.asarray(coupling_tensor)
    #         order = coupling_tensor.ndim
    #         if order == 0:
    #             continue
    #         spin_sign = (-1) ** order

    #         non_zero_indices = np.transpose(np.nonzero(coupling_tensor))
    #         for index_tuple in non_zero_indices:
    #             index_tuple = tuple(int(i) for i in index_tuple)

    #             # skip repeated indices
    #             if len(set(index_tuple)) != len(index_tuple):
    #                 continue
    #             # keep only strictly increasing tuples i1 < i2 < ... < ik
    #             if index_tuple != tuple(sorted(index_tuple)):
    #                 continue
    #             coeff = coupling_tensor[index_tuple]
    #             if coeff == 0:
    #                 continue

    #             term = qml.PauliZ(index_tuple[0])
    #             for q in index_tuple[1:]:
    #                 term = term @ qml.PauliZ(q)
    #             coeffs.append(sign * spin_sign * float(coeff))
    #             obs.append(term)

    #     return qml.Hamiltonian(coeffs, obs)

    def get_problem_hamiltonian(self, couplings, sign=1):
        """
        Construct Problem Hamiltonian from symmetric coupling tensors.
        Supports both 'ising' (-1/+1) and 'qubo' (0/1) input tensors.
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
                    

                elif self.model_type == "qubo":
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

    def get_mixer_hamiltonian(self, num_wires: int = None):
        """Constructs the Mixer Hamiltonian: Σ X_i (for a given full system or subgroup)"""
        if num_wires is None:
            num_wires = self.n_qubits
        return qml.Hamiltonian([1.0] * num_wires, [qml.PauliX(i) for i in range(num_wires)])

    def get_state_vector(self, s: str, gammas: float, time: float, gamma_mix: float) -> str:
        """
        
        Return the state vector.


        Parameters
        ----------
        s : str
            Input bitstring representing the initial state of the system.
        gammas : list of float
            List of coefficients for the input Hamiltonian terms. Length should match the number of coupling tensors in the model.
        time : float
            Total evolution time for the quantum circuit.
        gamma_mix : float
            Coefficient for the mixer Hamiltonian. Should be between 0 and 1.
        
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
        # Coefficients
        #alpha = self.model.calculate_alpha(couplings=self.local_couplings)
        #alpha = self.model.calculate_alpha(n=len(s), couplings=self.model.couplings) # dummy alpha for testing spectral gap, should not affect results
        self.gammas = gammas
        
        if gamma_mix < 0 or gamma_mix > 1:
            raise ValueError("gamma_mix must be between 0 and 1. Got gamma_mix:", gamma_mix)
        mixing_gamma = gamma_mix
        
        if np.any(np.array(self.gammas) < 0):
            raise ValueError("Gammas must be non-negative. Got gammas:", self.gammas)
        
        self.time = time
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))
        
        coeff_mixer = mixing_gamma
        coeff_problem = self.gammas#-(1 - self.gammas)

        # Do each hamiltonian term seperately, including those from each coupling tensor
        #H_total = qml.Hamiltonian([coeff_mixer] , [self.get_mixer_hamiltonian(num_wires)])
        H_total = qml.Hamiltonian([coeff_mixer] + list(np.ones(len(self.model.normalised_couplings))), [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=[self.model.normalised_couplings[i],], sign=coeff_problem[i]) for i in range(len(self.model.normalised_couplings))])
        #H_total = qml.Hamiltonian([coeff_mixer] + [1.,] , [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=self.model.normalised_couplings, sign=coeff_problem)])
        #H_total = qml.Hamiltonian([coeff_mixer] + [1.,] , [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=[self.model.normalised_couplings[0],], sign=coeff_problem[0])])
        @qml.qnode(dev)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, self.time, self.num_trotter_steps)
            return qml.state()

        state_vector = quantum_evolution(s)
        return state_vector

    # def get_sample_from_state_vector(self, s: str) -> str:
    #     """Returns a single sampled bitstring s' using the quantum distribution."""
    #     # Get the full state vector probabilities
    #     state_vector = self.get_state_vector(s)  # This returns the complex amplitudes
    #     probs = np.abs(state_vector) ** 2

    #     # Sample one index based on the probabilities
    #     n_states = len(probs)
    #     idx = np.random.choice(n_states, p=probs)

    #     # Convert that index back to a bitstring (e.g., 3 -> "011")
    #     s_prime = np.binary_repr(idx, width=self.model.n)
    #     return s_prime

    def get_sample(self, s_cg: str, time: float, gamma_mix: float, local_couplings: list, weights: float = None) -> str:
        
        """
        
        Returns a measured sample after time evolution


        Parameters
        ----------
        s : str
            Input bitstring representing the initial state of the system.
        time : float
            Total evolution time for the quantum circuit.
        gamma_mix : float
            Coefficient for the mixer Hamiltonian. Should be between 0 and 1.
        local_couplings : list of coupling tensors
            List of coupling tensors defining the problem Hamiltonian for the current subgroup.
        weights : list of float, optional
            List of coefficients for the input Hamiltonian terms. Length should match the number of coupling tensors in the model.
            Note that these weights are for the problem hamiltonian terms, and the mixer term is weighted separately by gamma_mix.
            Note that this weighting is optional, as somethings it may be required to weight the terms at a different point in the algorithm. See energy_model.get_subgroup_couplings for example of why we may need to apply the weighting at an unorthodox point.
            defaults to None, in which case no weighting is applied (equivalent to weights of 1 for all terms).
        
        Notes
        -----
        The total Hamiltonian simulated by the circuit is a weighted sum of the problem Hamiltonian terms and the mixer Hamiltonian:
        
        In Ferguson et al. (2025) [arXiv:2506.19538], we use gammas to weight the entire problem Hamiltonian vs the mixer vs the constraint Hamiltonian, such that the total Hamiltonian is:
        
        H = g_p * H_p + g_m * H_m + g_c * H_c

        but here we allow for separate gammas for each coupling tensor term, as well as a separate gamma for the mixer. The total Hamiltonian is then:

        H = (w_b1*H_b1+ w_b2*H_b2 +...+w_b2*H_bm) + g_m * H_m

        In other words, the constraint hamiltonian is absorbed in the coupling list, and weighted by the corresponding gamma in the gammas list. 
        This allows for more flexible weighting of different terms, and also allows us to use the same code for both constrained and unconstrained problems (by simply including or excluding the constraint Hamiltonian in the coupling list and adjusting the gammas accordingly).
        
        Note that it is assumed that each term is already normalised appropriately, so the gammas can be interpreted as the relative weights of each term in the total Hamiltonian.
        """
        if weights is None:
            weights = np.ones(len(local_couplings))


        num_wires = len(s_cg)
        dev = self._get_device(num_wires)
        
        # Coefficients
        # We need to be careful NIT to normalise again, as we have already added the coefficients

        #alphas = self.model.alphas#calculate_alpha(n=self.spin_length, couplings=self.local_couplings)
        # coeff_mixer = self.gamma
        # coeff_problem = -(1 - self.gamma)

        # # Do each hamiltonian term seperately, including those from each coupling tensor
        # H_total = qml.Hamiltonian([coeff_mixer] + list(np.ones(len(self.local_couplings))), [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=[self.local_couplings[i],], sign=coeff_problem) for i in range(len(self.local_couplings))])
        
        
        if gamma_mix < 0 or gamma_mix > 1:
            raise ValueError("gamma_mix must be between 0 and 1. Got gamma_mix:", gamma_mix)
        mixing_gamma = gamma_mix
        
        if np.any(np.array(weights) < 0):
            raise ValueError("Gammas must be non-negative. Got gammas:", weights)
        
        #self.time = time
        num_trotter_steps = int(np.floor((time / self.delta_time)))
        
        coeff_mixer = mixing_gamma
        coeff_problem = weights#-(1 - self.gammas)

        # Do each hamiltonian term seperately, including those from each coupling tensor
        #H_total = qml.Hamiltonian([coeff_mixer] + list(np.ones(len(self.model.normalised_couplings))), [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=[self.model.normalised_couplings[i],], sign=coeff_problem[i]) for i in range(len(self.model.normalised_couplings))])
        H_total = qml.Hamiltonian([coeff_mixer] + list(np.ones(len(local_couplings))), [self.get_mixer_hamiltonian(num_wires)]+ [self.get_problem_hamiltonian(couplings=[local_couplings[i],], sign=coeff_problem[i]) for i in range(len(local_couplings))])




        # set qnode to use our device with dynamically chosen wires
        @qml.qnode(dev, shots=1)
        def quantum_evolution(input_string):
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)
            qml.ApproxTimeEvolution(H_total, time, num_trotter_steps)
            return qml.sample()

        # Get the first shot from the sample
        sample = quantum_evolution(s_cg)[0]  # pennylane update doesnt squeeze singletons anymore
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring

    def update(self, s, subgroup_choice, local_couplings, gamma, time) -> str:
        """
        Performs time evolution on coarse grained hamiltonian update to get s' from s
        """
        
        self._assert_bitstring(s)

        # self.gamma = gamma
        # self.time = time
        # self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))
        # self.local_couplings = local_couplings
        #self.spin_length = len(subgroup_choice)
        
        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_sample(s_cg, time, gamma, local_couplings)

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
