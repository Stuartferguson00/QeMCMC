# External package imports
import numpy as np

# Internal package imports
from qemcmc.model.energy_model import EnergyModel


class ModelMaker:
    """
    Utility class for constructing standard Ising energy models used in simulations and experiments.

    This class constructs predefined random energy models used for testing
    and experimentation. Depending on the chosen ``model_type``, it generates
    coupling tensors and initialises an :class:`EnergyModel` instance.
    """

    def __init__(self, n_spins: int, model_type: str, name: str, cost_function_signs: list | None = None):
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

    

    def make_fully_connected_ising(self, return_couplings = False):
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        couplings = [h, J]
        self.model = EnergyModel(n=self.n_spins, couplings=couplings, name=self.name)
        if return_couplings:
            return couplings


    def make_fully_connected_binary(self):
        """
        Transforms the existing Ising couplings into an mathematically 
        equivalent QUBO model via s = 2x - 1.
        """
        couplings = self.make_fully_connected_ising(return_couplings=True)
        # all_energies_ising = self.model.get_all_energies()

        
        h,J = couplings

        #constant = 2*np.sum(J)-np.sum(h)
        Q_binary = 4 * J
        q_binary = 2 * h - 2 * np.sum(J, axis=1)
        binary_couplings = [np.round(q_binary, 4), np.round(Q_binary, 4)]
        
        self.model = EnergyModel(
            n=self.n_spins, 
            couplings=binary_couplings, 
            name=self.name, 
            cost_function_signs = [-1,-1],
            model_type="binary"
        )
        # all_energies_qubo = self.model.get_all_energies()
        # from matplotlib import pyplot as plt
        # plt.plot(np.arange(0, 2**self.n_spins,1), all_energies_qubo, label="QUBO")
        # plt.plot(np.arange(0, 2**self.n_spins,1), all_energies_ising, label="Ising")
        # plt.legend()
        # plt.show()
        # There appears to be a slight different in these. Not sure what it is, but I think its just a aconstant term issue
