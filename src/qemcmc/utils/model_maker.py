# Internal package imports
from qemcmc.model import EnergyModel
from qemcmc.coarse_grain import CoarseGraining

# External package imports
import numpy as np


class ModelMaker:
    # Class to control the initialisation of an energy model.
    # It might seem a bit convoluted, but will allow for more complex models to be made in future.
    def __init__(self, n_spins: int, model_type: str, name: str, J: np.ndarray = None, h: np.ndarray = None, cost_function_signs: list = [-1, -1]):
        self.name = name
        self.n_spins = n_spins
        self.cost_function_signs = cost_function_signs
        if type(model_type) is not str:
            print("model type must be a string representing the model you request")
        elif model_type == "Fully Connected Ising":
            self.make_fully_connected_Ising()
        elif model_type == "Fully Connected Ising Generic":
            self.make_fully_connected_Ising_generic()
        elif model_type == "Coarse Grained Ising":
            self.make_coarse_grained_ising()
        elif model_type == "1D Ising":
            self.make_1D_Ising()

    def make_fully_connected_Ising_generic(self):
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        couplings = [h, J]
        alpha = np.sqrt(self.n_spins) / np.sqrt(sum([J[i][j] ** 2 for i in range(self.n_spins) for j in range(i)]) + sum([h[j] ** 2 for j in range(self.n_spins)]))
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name, alpha=alpha)

    def make_coarse_grained_ising(self):
        # I define an energy model with the couplings and subgroups list explicitly specified in the parameters of EnergyModel object
        # This kind of initialization of an EnergyModel is required to run the CM.update() method for coarse graining
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        couplings = [h, J]
        alpha = np.sqrt(self.n_spins) / np.sqrt(sum([J[i][j] ** 2 for i in range(self.n_spins) for j in range(i)]) + sum([h[j] ** 2 for j in range(self.n_spins)]))
        # subgroups = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]  # [][[0, 1, 2], [3, 4, 5], [6]]  # Example of subgroups for coarse graining
        subgroups = [
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 5),
            (0, 1, 2, 3, 6),
            (0, 1, 2, 4, 5),
            (0, 1, 2, 4, 6),
            (0, 1, 2, 5, 6),
            (0, 1, 3, 4, 5),
            (0, 1, 3, 4, 6),
            (0, 1, 3, 5, 6),
            (0, 1, 4, 5, 6),
            (0, 2, 3, 4, 5),
            (0, 2, 3, 4, 6),
            (0, 2, 3, 5, 6),
            (0, 2, 4, 5, 6),
            (0, 3, 4, 5, 6),
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 6),
            (1, 2, 3, 5, 6),
            (1, 2, 4, 5, 6),
            (1, 3, 4, 5, 6),
            (2, 3, 4, 5, 6),
        ]
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name, alpha=alpha)
        self.cg = CoarseGraining(n=self.n_spins, subgroups=subgroups)
