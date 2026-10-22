from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from qemcmc.model.constraint_model import ConstraintModel
from qemcmc.model.energy_model import EnergyModel
from .basecircuitmaker import BaseCircuitMaker

class QuenchCircuitMaker(BaseCircuitMaker):
    """
        The total Hamiltonian simulated by the circuit is
    
            ``H = γ H_mixer + (1 - γ) α H_problem``
    
        where ``H_problem`` encodes the classical energy model and ``H_mixer``
        corresponds to a transverse-field term. The evolution time ``t`` and
        number of Trotter steps ``r`` are passed per call, giving an effective
        Trotter step size of ``δt = t / r``. The evolution is approximated
        using Trotterisation via ``qml.ApproxTimeEvolution``.
    """

    def __init__(self, model: EnergyModel):
        super().__init__(model)
        self.method = "quench_quantum"
    
    def get_state_vector(self, s: str, weights: list[float], time: float, r: int, mix_weight: float) -> np.ndarray:
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
        r : int
            Number of Trotter steps for the approximate time evolution.
        mix_weight : float
            Coefficient for the mixer Hamiltonian.

        Returns
        -------
        np.ndarray
            The final state vector.


        Notes
        -----
        The total Hamiltonian simulated by the circuit is a weighted sum of the problem Hamiltonian terms and the mixer Hamiltonian:

        In Ferguson et al. (2025) [arXiv:2506.19538], we use gammas to weight the entire problem Hamiltonian vs the mixer vs the constraint Hamiltonian, such that the total Hamiltonian is:

        ``H = g_p * H_p + g_m * H_m + g_c * H_c``

        but here we allow for separate weights for each coupling tensor term, as well as a separate gamma for the mixer. The total Hamiltonian is then:

        ``H = (w_b1*H_b1 + w_b2*H_b2 + ... + w_b2*H_bm) + g_m * H_m``

        In other words, the constraint hamiltonian is absorbed in the coupling list, and weighted by the corresponding gamma in the gammas list.
        This allows for more flexible weighting of different terms, and also allows us to use the same code for both constrained and unconstrained problems (by simply including or excluding the constraint Hamiltonian in the coupling list and adjusting the gammas accordingly).

        Note that it is assumed that each term is already normalised appropriately, so the gammas can be interpreted as the relative weights of each term in the total Hamiltonian.

        """

        num_wires = len(s)
        dev = self._get_device(num_wires)

        # if mix_weight < 0 or mix_weight > 1:
        #    raise ValueError(f"mix_weight must be between 0 and 1. Got {mix_weight}")

        # if np.any(np.array(weights) < 0):
        #    raise ValueError(f"Weights must be non-negative. Got {weights}")

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
            qml.ApproxTimeEvolution(H_total, time, r)
            return qml.state()

        state_vector = quantum_evolution(s)
        return state_vector

    def get_sample(self, s_cg: str, time: float, r: int, mix_weight: float, local_couplings: list, weights: list[float] | None = None) -> str:
        """
        Generate a single sample by evolving the system and measuring.

        Parameters
        ----------
        s_cg : str
            Input bitstring for the subgroup.
        time : float
            Total evolution time.
        r : int
            Number of Trotter steps for the approximate time evolution.
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

        # if mix_weight < 0 or mix_weight > 1:
        #    raise ValueError(f"mix_weight must be between 0 and 1. Got {mix_weight}")

        # if np.any(np.array(weights) < 0):
        #    raise ValueError(f"Weights must be non-negative. Got {weights}")

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
            qml.ApproxTimeEvolution(H_total, time, r)
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

        # This is slow!
        # Get the first shot from the sample
        # compiled_circuit = qml.compile(quantum_evolution)
        # sample = compiled_circuit(s_cg)[0]

        sample = quantum_evolution(s_cg)[0]
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring

    def update(self, s: str, subgroup_choice: list[int], local_couplings: list, gamma: float, time: float, r: int) -> str:
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
        r : int
            Number of Trotter steps for the approximate time evolution.

        Returns
        -------
        str
            The updated bitstring s'.
        """

        self._validate_bitstring(s)
        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_sample(s_cg, time, r, gamma, local_couplings)
        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            s_list[global_index] = s_cg_prime[i]
        return "".join(s_list)

