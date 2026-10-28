from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from qemcmc.model.constraint_model import ConstraintModel
from qemcmc.model.energy_model import EnergyModel

class BaseCircuitMaker:
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
    """
    

    def __init__(self, model: EnergyModel):
        self.model = model
        self.n_qubits = model.n
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)

        self.model_type = model.model_type
        self.devices = {}  # cache devices for dynamic subgroup sizes if needed

    def _get_device(self, num_wires: int):
        """Get or create a PennyLane device for the given number of wires."""
        if num_wires not in self.devices:
            self.devices[num_wires] = qml.device("lightning.qubit", wires=num_wires)
            # self.devices[num_wires] = qml.device("default.tensor", wires=num_wires, method="mps", max_bond_dim=2, contract="auto-mps")
        return self.devices[num_wires]

    def get_problem_hamiltonian(self, couplings: List[np.ndarray], sign: int = 1) -> qml.Hamiltonian:
        """
        Construct the problem Hamiltonian from symmetric coupling tensors.

        This method supports both 'ising' (-1/+1) and 'binary' (0/1) models.

        Parameters
        ----------
        couplings : List[dimod.BinaryPolynomial]
            A list of coupling polynomials.
        sign : int, optional
            A sign to apply to the Hamiltonian. Default is ``1``.

        Returns
        -------
        qml.Hamiltonian
            The problem Hamiltonian.
        """
        total_hamiltonian = 0.0 * qml.Identity(0)
        for coupling_poly in couplings:
            for indices, coeff in coupling_poly.items():
                if coeff == 0.0:
                    continue

                order = len(indices)
                if order == 0:
                    continue

                # Ensure strict increasing order to match previous behavior if needed,
                # though dimod already canonicalizes tuples.
                indices = tuple(sorted(indices))
                
                # Z |0> = |0>, Z |1> = - |1>. 
                # For ising models, bit=0 -> spin=-1, which corresponds to +1 Z eigenvalue. So spin_i = -Z_i.
                # Therefore a term with k spins has a factor of (-1)^k.
                spin_sign = (-1) ** order if self.model_type == "ising" else 1

                if self.model_type == "ising":
                    term = qml.PauliZ(indices[0])
                    for q in indices[1:]:
                        term = term @ qml.PauliZ(q)
                    total_hamiltonian += (sign * spin_sign * coeff) * term

                elif self.model_type == "binary":
                    # 0.5 * (I - Z) for first variable
                    #term = 0.5 * (qml.Identity(index_tuple[0]) + qml.PauliZ(index_tuple[0]))
                    term = 0.5 * (qml.Identity(indices[0]) - qml.PauliZ(indices[0]))
                    # multiply by 0.5 * (I - Z) for rest
                    for q in indices[1:]:
                        next_var = 0.5 * (qml.Identity(q) - qml.PauliZ(q))
                        #next_var = 0.5 * (qml.Identity(q) + qml.PauliZ(q))
                        term = term @ next_var

                    total_hamiltonian += (sign * coeff) * term
                    
        simplified_H = qml.simplify(total_hamiltonian)

        coeffs, ops = simplified_H.terms()
        return qml.Hamiltonian(coeffs, ops)

    def get_mixer_hamiltonian(self, num_wires: int | None = None) -> qml.Hamiltonian:
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
    def _validate_bitstring(self, s: str, *, length: int | None = None):
        """
        Validate s is a bitstring.

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

    def check_Hamiltonian(self, basis_states_ints: list[str] | None = None, couplings: list[np.ndarray] | None = None, plot=False) -> None:
        """
        Utility function to print the Energy of the Hamiltonian to be simulated. Naturally, for a classical problem, the energy of the Hamiltonian wrt to some basis state b H|b> = E|b> should equal the energy of the corresponding state in the classical model. This is a useful sanity check to ensure that the Hamiltonian is being constructed correctly from the coupling tensors.


        """

        # Get hamiltonian (not normalised)
        Hamiltonian = self.get_problem_hamiltonian(couplings)
        energies = []
        quantum_energies = []
        # Evaluate the Hamiltonian on some randomly chosen bitstring states, and compare to the classical energy of those states in the model
        for state in basis_states_ints:
            print("state int: ", state)
            state = bin(state)[2:].zfill(self.n_qubits)
            classical_energy = self.model.get_energy(state)

            dev = qml.device("lightning.qubit", wires=self.n_qubits)

            @qml.qnode(dev)
            def evaluate_energy(basis_state):
                # Prepares the state |1, 0> if basis_state=[1, 0]
                # print(basis_state)
                # print(len(bin(state)[2:].zfill(self.n_qubits)))
                # qml.BasisState(, wires=range(self.n_qubits))

                for i, bit in enumerate(basis_state):
                    if bit == "1":
                        qml.PauliX(i)
                return qml.expval(Hamiltonian)

            quantum_energy = evaluate_energy(state)

            if type(self.model) is ConstraintModel:
                constraint_energy = self.model.get_constraint_energy(state)
                print(
                    f"State: {state}, Classical Energy: {np.round(classical_energy, 2)}, Quantum Hamiltonian Energy: {np.round(quantum_energy, 2)}, Constraint energu {np.round(constraint_energy, 2)},Constraint Satisfaction: {self.model.constraint_func(state)}"
                )
            else:
                print(f"State: {state}, Classical Energy: {np.round(classical_energy, 2)}, Quantum Hamiltonian Energy: {np.round(quantum_energy, 2)}")
            energies.append(classical_energy)
            quantum_energies.append(quantum_energy)
        if plot:
            plt.scatter(energies, quantum_energies)
            plt.xlabel("Classical Energy")
            plt.ylabel("Quantum Energy")
            plt.show()
            plt.plot(np.arange(0, 2**self.n_qubits), np.array(energies), label="classical")
            plt.plot(np.arange(0, 2**self.n_qubits), np.array(quantum_energies), label="quantum")
            plt.show()


