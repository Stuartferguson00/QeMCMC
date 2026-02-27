# Internal package imports
from qemcmc.model import EnergyModel
from qemcmc.coarse_grain import CoarseGraining

# External package imports
import numpy as np
import itertools


class ModelMaker:
    # Class to control the initialisation of an energy model.
    # It might seem a bit convoluted, but will allow for more complex models to be made in future.
    def __init__(self, n_spins: int, model_type: str, name: str, coarse_graining_number: int = 1, cost_function_signs: list = [-1, -1]):
        self.name = name
        self.n_spins = n_spins
        self.cost_function_signs = cost_function_signs or [-1, -1]

        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")

        if model_type == "Fully Connected Ising":
            self.make_fully_connected_ising()
        elif model_type == "Coarse Grained Ising":
            self.make_coarse_grained_ising(coarse_graining_number)
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

    def make_coarse_grained_ising(self, coarse_graining_number):
        # I define an energy model with the couplings and subgroups list explicitly specified in the parameters of EnergyModel object
        # This kind of initialization of an EnergyModel is required to run the CM.update() method for coarse graining
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        couplings = [h, J]
        subgroups = list(itertools.combinations(range(self.n_spins), coarse_graining_number))
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name)
        self.cg = CoarseGraining(n=self.n_spins, subgroups=subgroups)
