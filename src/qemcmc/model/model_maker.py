# Internal package imports
from qemcmc.model import EnergyModel
from qemcmc.coarse_grain import CoarseGraining
from typing import List

# External package imports
import numpy as np
import itertools
import dimod


class FastIsingEnergy:
    """
    Globally accessible callable to calculate energy.
    Extracting this from ModelMaker allows it to be safely pickled for multiprocessing.
    """
    def __init__(self, h_np: np.ndarray, J_np: np.ndarray):
        self.h_np = h_np
        self.J_np = J_np

    def __call__(self, state):
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        
        state_arr = np.array([int(bit) for bit in state])
        state_arr = (state_arr << 1) - 1

        energy = - np.dot(state_arr, self.h_np) - np.dot(state_arr, self.J_np @ state_arr) / 2
        return energy


class ModelMaker:
    """
    Utility class for constructing standard Ising energy models used in simulations and experiments.
    """
    def __init__(self, n_spins: int, model_type: str, name: str, cost_function_signs: list = None):
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

        # Replaced the nested function with the picklable callable class
        get_energy_manual = FastIsingEnergy(h_np, J_np)

        couplings = [h, J]
        self.model = EnergyModel(
            n=self.n_spins, 
            couplings=couplings, 
            name=self.name, 
            cost_function_signs=self.cost_function_signs, 
            model_type="ising", 
            manual_get_energy=get_energy_manual
        )
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
