import numpy as np
import itertools
import typing
import abc
from typing import List, Union


class EnergyModel(abc.ABC):
    """
    Abstract base class for energy models. Initializes with a couplings list.
    """
    def __init__(self, n: int, couplings: List[np.ndarray] = [], name: str = None) -> None:
        """
        Initialize an energy model. This acts as a base class for specific energy models (such as Ising models).
        Parameters:
            n (int): Number of spins in the model.
            couplings (List[np.ndarray]): List of numpy arrays representing coupling tensors.
            name (str, optional): Name of the model. Defaults to None.
            """
        self.n = n
        self.couplings = couplings
        self.name = name


    @abc.abstractmethod
    def calc_an_energy(self, state: str) -> float:
        """
        Calculate the energy of a given state.
        This function is Model-specific and is expected to be implemented in subclasses.

        Args:
        state (str): A string representing the state, where each character is either "0" or "1".
        
        Returns
        float: The calculated energy of the given state.
        """
        pass
    
    def calculate_energy_from_couplings(self, state: Union[str, List[int]], state_representation: str = 'spin', sign: int = 1) -> float:
        """ 
        Calculates the energy for any arbitrary-order Ising/QUBO model.
        Args:
            couplings: An unordered list of numpy arrays representing
                all coupling terms. The function determines the interaction order from
                each array's dimension (e.g., 1D for fields, 2D for interactions).

            state: The configuration state.
                - If 'spin', provide a list of ints, e.g., [1, -1, 1].
                - If 'binary', provide a string of '0's and '1's, e.g., '101'.

            state_representation: The convention for the state variables.
                Defaults to 'spin'.

            sign: The sign convention for the coupling terms.
                Defaults to 1.

        Returns:
            float: The total calculated energy of the state.
        """
        couplings = self.couplings

        if state_representation == 'spin':
            if not isinstance(state, list):
                raise TypeError("For 'spin' format, the state must be a list of integers (e.g., [1, -1, 1]).")
            s = np.array(state, dtype=np.int8)
            if not np.all(np.isin(s, [-1, 1])):
                 raise ValueError("Spin states in list format must only contain 1 and -1.")
        
        elif state_representation == 'binary':
            if not isinstance(state, str):
                raise TypeError("For 'binary' format, the state must be a string (e.g., '101').")
            # Convert the binary string '0'/'1' to a spin array [-1, 1] for calculation.
            s = np.array([2 * int(char) - 1 for char in state], dtype=np.int8)
        
        else:
            raise ValueError("state_representation must be either 'spin' or 'binary'.")

        total_energy = 0.0
        einsum_indices = 'ijklmnopqrstuvwxyz' # Provides indices for up to 26-order interactions
        num_spins = len(s)

        # Iterate through the coupling tensors (h, J, L, ...)
        for coeffs in couplings:
            # Determine the interaction order from the array's dimension
            order = coeffs.ndim
            if order == 0:
                total_energy += coeffs.item()
                continue
            # Validate that the tensor dimensions match the number of spins
            if not all(dim == num_spins for dim in coeffs.shape):
                raise ValueError(f"Shape {coeffs.shape} of a coupling tensor is not "
                                 f"compatible with the number of spins ({num_spins}).")
            if order > len(einsum_indices):
                raise ValueError(f"Cannot handle interaction order > {len(einsum_indices)}.")

            # Dynamically build and execute the einsum calculation for this term
            # For order=1 (h): 'i,i->' sums h_i * s_i
            # For order=2 (J): 'i,j,ij->' sums J_ij * s_i * s_j
            # ... etc.
            indices = einsum_indices[:order]
            path = ','.join(indices) + ',' + indices + '->'
            operands = [s] * order + [coeffs]
            total_energy += np.einsum(path, *operands)

        return sign * total_energy
    
    
    def get_energy(self, state: str) -> float:
        """ 
        Returns the energy of a given state
        """
        if not isinstance(state,str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        
        energy = self.calc_an_energy(state)
        
        return energy
    
    def get_all_energies(self) -> np.ndarray :
        """
        Calculate the energies for all possible spin states.
        This method generates all possible spin states for the system, calculates the energy for each state,
        and returns an array of these energies.
        Returns:
            np.ndarray: An array containing the energies of all possible spin states.
        """
        self.S = [''.join(i) for i in itertools.product('01', repeat=self.num_spins)]
        all_energies = np.zeros(len(self.S))
        for state in self.S:
            all_energies[int(state,2)] = self.calc_an_energy(state)
        return all_energies
            
            
    def get_lowest_energies(self, num_states:int) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the lowest energy states and their degeneracies.
        This method computes all possible energies and then finds the specified number
        of lowest energy states along with their degeneracies. Note that this method 
        is intended for small instances due to its brute-force nature, which is extremely 
        memory intensive and slow.
        Args:
            num_states (int): The number of lowest energy states to retrieve.
        Returns:
            Two numpy arrays:
                - The first array contains the lowest energy values.
                - The second array contains the degeneracies of the corresponding energy values.
        """
        #only to be used for small instances, it is just brute force so extremely memory intensive and slow
        all_energies = self.get_all_energies()

        #very slow (sorts whole array)  
        self.lowest_energies, self.lowest_energy_degeneracy = self.find_lowest_values(all_energies, num_values=num_states)

        return self.lowest_energies, self.lowest_energy_degeneracy
    
    def find_lowest_values(self, arr: np.ndarray, num_values: int = 5):
        """
        Find the lowest unique values in an array and their degeneracies.

        Args:
            arr (np.ndarray): The input array from which to find the lowest values.
            num_values (int, optional): The number of lowest unique values to find. Defaults to 5.

        Returns:
        tuple: A tuple containing two numpy arrays:
            - lowest_values (np.ndarray): The lowest unique values in the array.
            - degeneracy (np.ndarray): The counts of each of the lowest unique values.
        """
        # Count the occurrences of each value
        unique_values, counts = np.unique(arr, return_counts=True)
        # Sort the unique values and counts by value
        sorted_indices = np.argsort(unique_values)
        unique_values_sorted = unique_values[sorted_indices]
        counts_sorted = counts[sorted_indices]
        # Find the first num_values
        lowest_values = unique_values_sorted[:num_values]
        degeneracy = counts_sorted[:num_values]
        return lowest_values, degeneracy
    
    def get_lowest_energy(self):
        """
        Calculate and return the lowest energy from all possible energies.
        This method uses a brute force approach to find the lowest energy, 
        making it extremely memory intensive and slow. It is recommended 
        to use this method only for small instances.
        Returns:
            float: The lowest energy value.
        Notes:
            If the lowest energy has already been calculated and stored 
            in `self.lowest_energy`, it will return that value directly 
            to save computation time.
        """
        
        
        # Only to be used for small instances, it is just brute force so extremely memory intensive and slow
        if self.lowest_energy is not None:
            return self.lowest_energy
        else:
            all_energies = self.get_all_energies()

        lowest_energy = np.min(all_energies)

        return lowest_energy

    def get_boltzmann_factor(self, state: str, beta: float = 1.0) -> float:
        """ 
        Get un-normalised boltzmann probability of a given state 

        Args:
            state (str): configuration of spins for which probability is to be calculated 
            beta (float): inverse temperature (1/T) at which the probability is to be calculated.
        
        Returns:
            float corresponding to the un-normalised boltzmann probability of the given state.
        """
        E = self.get_energy(state)
        r = np.exp(-1 * beta * E, dtype = np.longdouble)

        return r

    
    def get_boltzmann_factor_from_energy(self, E, beta: float = 1.0
    ) -> float:

        """
        Get un-normalized Boltzmann probability for a given energy.

        Args:
            E (float): Energy for which the Boltzmann factor is to be calculated.
            beta (float): Inverse temperature (1/T) at which the probability is to be calculated.

        Returns:
            float: The un-normalized Boltzmann probability for the given energy.
        """
        
        return np.exp(-1 * beta * E, dtype = np.longdouble)
    


class IsingEnergyFunction(EnergyModel):
    """
    A class to build the Ising Energy Function from self.
    Heavily modified from https://github.com/pafloxy/quMCMC to add functionality.
    Attributes:
    -----------
    negative_energy : bool
        Indicates if the energy should be negative.
    J : np.array
        Weight-matrix of the interactions between the spins.
    h : np.array
        Local field to the spins.
    S : list or None
        List of all possible states.
    all_energies : np.array or None
        Array of energies for all possible states.
    lowest_energy : float or None
        The lowest energy found.
    num_spins : int
        Number of spins in the system.
    alpha : float
        Scaling factor for the energy.
    name : str
        Name of the Ising model.
    initial_state : list
        List of initial states for simulations.
    Methods:
    --------
    __init__(J: np.array, h: np.array, name: str = None, negative_energy: bool = True, no_inits = False) -> None
        Initializes the IsingEnergyFunction with given parameters.
    get_energy(state: str) -> float
        Returns the energy of a given state.
    calc_an_energy(state: str) -> float
        Calculates the energy of a given state.
    get_all_energies() -> None
        Calculates and stores the energies of all possible states.
    get_lowest_energies(num_states: int) -> tuple[np.array, np.array]
        Returns the lowest energies and their degeneracies.
    find_lowest_values(arr: np.array, num_values: int = 5) -> tuple[np.array, np.array]
        Finds the lowest values in an array and their counts.
    get_lowest_energy() -> float
        Returns the lowest energy found.
    get_boltzmann_factor(state: Union[str, np.array], beta: float = 1.0) -> float
        Returns the un-normalized Boltzmann probability of a given state.
    get_boltzmann_factor_from_energy(E: float, beta: float = 1.0) -> float
        Returns the un-normalized Boltzmann probability for a given energy.
    """    

    def __init__(self, J: np.array, h: np.array, name:str = None, cost_function_signs:list = [-1,-1], no_initial_states = False) -> None:
        """
        Initialize the Ising model.
        Parameters:
            J (np.array): Weight-matrix of the interactions between the spins.
            h (np.array): Local field to the spins.
            name (str, optional): Name of the Ising model. Defaults to None.
            cost_function_signs (list, optional): List of two elements, the first element is the sign of the interaction term and the second element is the sign of the field term. Allows for the cost function to be flipped with respect to the standard Ising model. Defaults to [-1, -1].
            no_initial_states (bool, optional): If True, no initial states are stored for the model, randomly generated. Defaults to False.
        """
        super().__init__(n = J.shape[0], name = name)
        
        # self.cost_function_signs allows for cost function to be flipped wrt to the standard Ising model
        self.cost_function_signs = cost_function_signs
        self.J = J
        self.h = h
        self.S = None
        self.lowest_energy  = None
        self.num_spins =self.n
        self.alpha = np.sqrt(self.num_spins) / np.sqrt( sum([J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([h[j]**2 for j in range(self.num_spins)])  )
        if no_initial_states:
            self.initial_state = []
        else:
            self.initial_state = []
            for i in range(100): 
                self.initial_state.append(''.join(str(i) for i in np.random.randint(0, 2, self.n, dtype = int)))

    
            
    def calc_an_energy(self, state:str) -> float:
        """
        Calculate the energy of a given state.
        
        This function computes the energy of a given state based on the Ising model.
        The state is expected to be a string of "0"s and "1"s, which are converted to
        -1 and 1 respectively for the calculation.
        
        Args:
        state (str): A string representing the state, where each character is either "0" or "1".
        
        Returns
        float: The calculated energy of the given state.
        
        Raises:
        TypeError
            If the input state is not a string.
        """

        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        
        state = np.array([-1 if elem == "0" else 1 for elem in state])
        
        # THIS ONLY WORKS IF THE INPUT IS NOT UPPER DIAGONAL.
        # self.cost_function_signs allows for cost function to be flipped wrt to the standard Ising model
        try:
            energy = self.cost_function_signs[0]* 0.5 * np.dot(state.transpose(), self.J.dot(state)) + self.cost_function_signs[1]* np.dot(self.h.transpose(), state)
        except Exception as e:
            print(f"Error calculating energy for state {state}: {e}")
            print("This error is generally caused when qulacs outputs a bitstring of 1 followed by n 0's for the state for some reason")
            energy = 10000

        return energy
    
    
    @property
    def get_J(self):
        return self.J
    
    @property
    def get_h(self):
        return self.h




# if __name__ == "__main__":

#     # Linear coefficients (h vector)
#     h = np.array([-1.0, -2.0, -3.0])

#     # Quadratic coefficients (J matrix)
#     # J_12 = 0.5, J_23 = -1.5
#     J = np.array([
#         [0.0, 0.5, 0.0],
#         [0.5, 0.0, -1.5],
#         [0.0, -1.5, 0.0]
#     ])

#     # Create the list of coupling tensors
#     my_couplings = [h, 0.5 * J]

#     my_state = '011'

#     energies = []
#     energies2 = []
#     for state in ['000', '001', '010', '011', '100', '101', '110', '111']:

#         print("Calculating energy for state:", state)
#         energy_model = EnergyModel(n=3, couplings=my_couplings)
#         energy = energy_model.calculate_energy_from_couplings(state=state, state_representation='binary', sign=-1)
#         print("Energy using couplings:", energy)
#         energies.append(energy)

#         ising_model = IsingEnergyFunction(J=J, h=h, name="Test Ising Model")
#         energy_ising = ising_model.get_energy(state)
#         print("Energy using IsingEnergyFunction:", energy_ising)
#         energies2.append(energy_ising)

#         print("\n\n")

#     print("Lowest energy is", min(energies))
#     print("Lowest energy (Ising model) is", min(energies2))