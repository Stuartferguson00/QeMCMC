# Internal package imports
from qemcmc.model import EnergyModel
from qemcmc.coarse_grain import CoarseGraining
from typing import List
# External package imports
import numpy as np
import itertools
import dimod


class ModelMaker:
    """
    Utility class for constructing standard Ising energy models used in simulations and experiments.

    This class constructs predefined random energy models used for testing
    and experimentation. Depending on the chosen ``model_type``, it generates
    coupling tensors and initialises an :class:`EnergyModel` instance.
    """

    def __init__(self, n_spins: int, model_type: str, name: str, cost_function_signs: list = [-1, -1]):
        self.name = name
        self.n_spins = n_spins
        self.cost_function_signs = cost_function_signs or [-1, -1]

        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")

        if model_type == "Fully Connected Ising":
            self.make_fully_connected_ising()
        elif model_type == "Fully Connected QUBO":
            self.make_fully_connected_binary()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def make_fully_connected_ising(self, return_couplings=False):
        shape_of_J = (self.n_spins, self.n_spins)
        J_np = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_np = np.tril(J_np, -1) + np.tril(J_np, -1).transpose()
        h_np = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        h_dict = {(i,): float(h_np[i]) for i in range(self.n_spins) if h_np[i] != 0}
        J_dict = {(i, j): float(J_np[i, j]) for i in range(self.n_spins) for j in range(i + 1, self.n_spins) if J_np[i, j] != 0}

        h = dimod.BinaryPolynomial(h_dict, dimod.SPIN)
        J = dimod.BinaryPolynomial(J_dict, dimod.SPIN)


        # dimod representation needed for sparsity etc. 
        # but a simple numpy implementation is far more efficient for energy calc.

        def get_energy_manual(state):
            """
            Calculate the energy of a given state for an arbitrary-order Ising/binary model.

            Parameters
            -----------
            state : array-like (str, list, tuple, np.array)

                State configuration. Can be:
                - Binary: "011", [0,1,1], (0,1,1), etc.
                - Spin: [-1,1,1], (-1,1,1), etc.

            couplings : list of numpy arrays
            
            List of coupling tensors where:
                - 1D arrays represent linear terms (h_i)
                - 2D arrays represent quadratic terms (J_ij)
                - 3D arrays represent cubic terms, etc.

            Returns
            -------
            float : Total energy of the state
            """
            if not isinstance(state, str):
                raise TypeError(f"State must be a string, but got {type(state)}")
            
            state = np.array([int(bit) for bit in state])
            state = (state << 1) - 1

            energy = - np.dot(state, h_np) - np.dot(state, J_np @ state)/2
            return energy

        couplings = [h, J]
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name, cost_function_signs=self.cost_function_signs, model_type="ising", manual_get_energy = get_energy_manual)
        if return_couplings:
            return couplings

    def make_fully_connected_binary(self):
        """
        Transforms the existing Ising couplings into an mathematically 
        equivalent QUBO model via s = 2x - 1.
        """
        shape_of_J = (self.n_spins, self.n_spins)
        J_np = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_np = np.tril(J_np, -1) + np.tril(J_np, -1).transpose()
        h_np = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        Q_binary_np = 4 * J_np
        q_binary_np = 2 * h_np - 2 * np.sum(J_np, axis=1)

        q_dict = {(i,): float(np.round(q_binary_np[i], 4)) for i in range(self.n_spins) if q_binary_np[i] != 0}
        Q_dict = {(i, j): float(np.round(Q_binary_np[i, j], 4)) for i in range(self.n_spins) for j in range(i + 1, self.n_spins) if Q_binary_np[i, j] != 0}

        q_binary = dimod.BinaryPolynomial(q_dict, dimod.BINARY)
        Q_binary = dimod.BinaryPolynomial(Q_dict, dimod.BINARY)

        binary_couplings = [q_binary, Q_binary]
        
        self.model = EnergyModel(
            n=self.n_spins, 
            couplings=binary_couplings, 
            name=self.name, 
            cost_function_signs=[-1, -1],
            model_type="binary"
        )
