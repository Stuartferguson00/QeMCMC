# Internal package imports
from qemcmc.model import EnergyModel
from qemcmc.coarse_grain import CoarseGraining

# External package imports
import numpy as np
import itertools


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
        elif model_type == "1D Ising":
            self.make_1D_Ising()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def make_fully_connected_ising(self):
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        couplings = [h, J]
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name)

    