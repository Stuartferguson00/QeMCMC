
import itertools
from operator import pos
import matplotlib.pyplot as plt

import numpy as np
from itertools import combinations, permutations, product
import math
from typing import List, Tuple, Union, Dict
import os
import pickle
import numpy.typing as npt
from tqdm import tqdm

from qemcmc.utils.helpers import MCMCChain

def calc_interval_abundances(causal_matrix: npt.NDArray ) -> npt.NDArray[np.int32]:
    """
    Calculate interval abundances based on a given causal matrix.
    This function computes the relative abundances of intervals in a causal matrix.
    
    Args:
        causal_matrix (numpy.ndarray): A square matrix representing causal relationships.
    
    Returns:
        numpy.ndarray: An array of relative abundances of intervals, the first element
                    is the cardinality of the causal set, and subsequent elements 
                    represent the number of IOIs (see cunningham 2018)
    """

    n = causal_matrix.shape[0]
    adj_mat = causal_matrix
    past_mat = causal_matrix.T

    rel_abundances = np.zeros(n+1)
    # Loop through all pairs
    for i, val_i in enumerate(adj_mat):
        for j, val_j in enumerate(adj_mat):
            # trivial way to find cardinality
            if i == j:
                rel_abundances[0] += 1
            
            # if i is in the past of j, count the number of elements in interval
            elif adj_mat[i,j] ==1:
                #number of matching 1's in row of adj matrix
                #from Naive action algorithm in cunningham 2018
                #only want to search in the future of j... so only top (or bottom depending on setup) half of matrix
                k = len(np.where((adj_mat[i,:]==past_mat[j,:])&(adj_mat[i,:] ==1))[0])
                rel_abundances[k+1] +=1
    # Return as integer array
    return rel_abundances.astype(np.int32)

def calculate_action(causal_matrix: npt.NDArray, smeared: bool = True, stdim: int = 2, epsilon: float = 0.1, first_order_smearing: bool = False, first_order_taylor: bool = False):
    """
        Calculate the action based on the given causal matrix and parameters.
        
        Args:
            causal_matrix (numpy.ndarray): The causal matrix used to calculate interval abundances.
            smeared (bool, optional): Whether to apply smearing. Default is True.
            stdim (int, optional): The spacetime dimension. Default is 2. (other dimensions not yet implimented)
            epsilon (float, optional): The epsilon parameter for smearing. Default is 0.1.
            first_order_smearing (bool, optional): Whether to use first-order smearing. Default is False.
            first_order_taylor (bool, optional): Whether to use first-order Taylor expansion. Default is False. Infers first_order_smearing=True.
        
        Returns:
            float: The calculated action.
        """
    
    #if stdim != 2:
    #    raise NotImplementedError("Only 2D is currently implemented")
    
    if first_order_taylor:
        if first_order_smearing == False:
            print("Assuming first_order_smearing=True due to first_order_taylor=True")
            first_order_smearing = True
    
    
    c = calc_interval_abundances(causal_matrix)
    a = 0
    
    
    if smeared:
        if first_order_smearing:
            if first_order_taylor:
                eps1 = epsilon / (1.0 - epsilon)
                for i in range(0, c[0] - 1):
                    ni = float(c[i + 1])
                    if stdim == 2:
                        a += ni * (1.0-i*epsilon)
                    elif stdim == 4:
                        a += ni * (1.0-i*epsilon)
                    #print("i: ", i)
                    #print("a contribution from i: ", ni * (1.0-i*epsilon))

                #print("a before factor: ", a)
                if stdim == 2:
                    a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
                elif stdim == 4:
                    a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
            else:
                eps1 = epsilon / (1.0 - epsilon)
                for i in range(0, c[0] - 1):
                    ni = float(c[i + 1])
                    if stdim == 2:
                        a += ni * math.pow(1.0 - epsilon, i) 
                    elif stdim == 4:
                        a += ni * math.pow(1.0 - epsilon, i) 

                if stdim == 2:
                    a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
                elif stdim == 4:
                    a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
            
        else:               
            eps1 = epsilon / (1.0 - epsilon)
            for i in range(0, c[0] - 1):
                ni = float(c[i + 1])
                if stdim == 2:
                    a += ni * math.pow(1.0 - epsilon, i) * (1.0 - 2.0 * eps1 * i + 0.5 * eps1 * eps1 * i * (i - 1.0))
                elif stdim == 4:
                    a += ni * math.pow(1.0 - epsilon, i) * (1.0 - 9.0 * eps1 * i + 8.0 * eps1 * eps1 * i * (i - 1.0) - (4.0 / 3.0) * eps1 * eps1 * eps1 * i * (i - 1.0) * (i - 2.0))

            if stdim == 2:
                a= 2.0 * epsilon * (c[0] - 2.0 * epsilon * a)
            elif stdim == 4:
                a= (4.0 / math.sqrt(6.0)) * (math.sqrt(epsilon) * c[0] - math.pow(epsilon, 1.5) * a)
    else:
        if stdim == 2:
            a = 2.0 * (c[0] - 2.0 * (c[1] - 2.0 * c[2] + c[3]))
        elif stdim == 4:
            a = (4.0 / math.sqrt(6.0)) * (c[0] - c[1] + 9.0 * c[2] - 16.0 * c[3] + 8.0 * c[4])
            
    return a



def transitive_closure(a:npt.NDArray) -> npt.NDArray:
    """
    Computes the transitive closure of a given adjacency matrix using Warshall's algorithm.
    The transitive closure of a graph is a reachability matrix that indicates whether there is a path 
    between any pair of vertices in the graph.
    Taken from https://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
    Parameters:
    a (numpy.ndarray): A square adjacency matrix representing the graph. The matrix should be a 2D numpy array 
        where a[i][j] is True if there is an edge from vertex i to vertex j, and False otherwise.
    Returns:
    numpy.ndarray: A square matrix of the same size as the input matrix, where the element at position (i, j) 
        is True if there is a path from vertex i to vertex j, and False otherwise.
    """
    
    
    
    
    n = len(a)
    m = np.copy(a)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                m[i][j] = m[i][j] or (m[i][k] and m[k][j])
    return m


def transitive_reduction(a):
    
    # need to first close it... ugh
    m = transitive_closure(a)
    n = a.shape[0]
    for j in range(n):
        for i in range(n):
            if (m[i][j]):
                for k in range(n):
                    if (m[j][k]):
                        m[i][k] = 0
    return m
    
    
def is_causal_matrix(matrix:npt.NDArray) -> bool:
    """
    Checks if a given upper triangular binary matrix corresponds to the causal matrix of a causal set.
    
    Parameters:
    matrix (np.ndarray): The upper triangular binary matrix to check.
    
    Returns:
    bool: True if the matrix corresponds to a causal matrix, False otherwise.
    """
    
    m = transitive_closure(matrix)
    if np.all(m == matrix):
        return True
    else:
        return False

def get_unique_matrices(n:int) -> Tuple[set, set]:
    """
    Generate unique matrices and unique causal matrices of size n x n.
    This function generates all possible unique upper triangular binary matrices 
    of size n x n, and then filters out those that are causal matrices.
    
    Parameters:
    n (int): The size of the matrices to generate.
    
    Returns:
    Tuple[set, set]: A tuple containing two sets:
        - The first set contains all unique upper triangular binary matrices.
        - The second set contains all unique causal matrices.
    """
    
    
    # Create the directory if it doesn't exist
    filepath = os.path.dirname(__file__)
    save_folder = "save_files"
    save_path = os.path.join(filepath, save_folder)
    
    
    try:
        unique_matrices = pickle.load(open(os.path.join(save_path, f"unique_matrices_"+str(n)+".pkl"), "rb"))
        unique_causal_matrix = pickle.load(open(os.path.join(save_path, f"unique_causal_matrices_"+str(n)+".pkl"), "rb"))
    except:
        num_unique = 2**((n**2-n)//2)
        
        
        unique_matrices = set()
        for bits in tqdm(product([0, 1], repeat=(n * (n - 1)) // 2), total=num_unique, desc=f"Generating unique matrices for n={n}"):
            matrix = np.zeros((n, n), dtype = np.int32)
            upper_tri_indices = np.triu_indices(n, 1)
            matrix[upper_tri_indices] = bits
            unique_matrices.add(matrix.tobytes())
            dtype_ = matrix.dtype

            
        #print(f"Number of unique matrices: {len(unique_matrices)}")
        if len(unique_matrices) != num_unique:
            raise ValueError(f"Number of unique matrices is not correct. Expected {num_unique}, got {len(unique_matrices)}")


        unique_causal_matrix = set()



        for unique_matrix in tqdm(unique_matrices, desc=f"Checking causal matrices for n={n}"):
            matrix = np.frombuffer(unique_matrix, dtype = dtype_).reshape(n,n)
            if is_causal_matrix(matrix):
                unique_causal_matrix.add(unique_matrix)


        unique_matrices = unique_matrices
        unique_causal_matrix = unique_causal_matrix
        print(f"Number of unique matrices: {len(unique_matrices)}")
        print(f"Number of unique causal matrices: {len(unique_causal_matrix)}")

        with open(os.path.join(save_path, f"unique_matrices_"+str(n)+".pkl"), "wb") as f:
            pickle.dump(unique_matrices, f)

        with open(os.path.join(save_path, f"unique_causal_matrices_"+str(n)+".pkl"), "wb") as f:
            pickle.dump(unique_causal_matrix, f)
        
    unique_matrices = sorted(unique_matrices, key=lambda x: np.frombuffer(x, dtype=np.int32).reshape(n, n).tolist())
    unique_causal_matrix = sorted(unique_causal_matrix, key=lambda x: np.frombuffer(x, dtype=np.int32).reshape(n, n).tolist())
    
    #print("Unique matrices: ", unique_matrices)
    #print("Unique causal matrices: ", unique_causal_matrix)
    return unique_matrices, unique_causal_matrix

def get_unique_causal_bitstrings(n:int) -> Tuple[npt.NDArray,npt.NDArray]:
    
    unique_matrices, unique_causal_matrices = get_unique_matrices(n)
    
    # Convert unique causal matrices into bitstring representations
    unique_bitstring_causal_matrices = []
    for string in unique_causal_matrices:
        matrix = np.frombuffer(string, dtype=np.int32).reshape((n, n))
        bitstring = ''.join(str(int(matrix[i, j])) for i in range(n) for j in range(i + 1, n))
        unique_bitstring_causal_matrices.append(bitstring)

    # Convert to a numpy array
    unique_bitstring_causal_matrices = np.array(unique_bitstring_causal_matrices)
    
    unique_bitstring_matrices = []
    # Convert unique causal matrices into bitstring representations
    for string in unique_matrices:
        matrix = np.frombuffer(string, dtype=np.int32).reshape((n, n))
        bitstring = ''.join(str(int(matrix[i, j])) for i in range(n) for j in range(i + 1, n))
        unique_bitstring_matrices.append(bitstring)
    
    unique_bitstring_matrices = np.array(unique_bitstring_matrices)
    
    
        # Sort full list of matrices
    arg_sorted = np.array(np.argsort(list(unique_bitstring_matrices), axis=0), dtype=int)
    unique_bitstring_matrices = np.array(list(unique_bitstring_matrices))[arg_sorted]

    # Sort causal matrices
    arg_sorted_causal = np.array(np.argsort(list(unique_bitstring_causal_matrices), axis=0), dtype=int)
    unique_bitstring_causal_matrices = np.array(list(unique_bitstring_causal_matrices))[arg_sorted_causal]
    
    return unique_bitstring_matrices, unique_bitstring_causal_matrices




def calculate_average_action(cardinality:int, causal_matrices: set, stdim: int = 2, epsilon: float = 0.1, Temp: float = 1) -> float:
    """
    Calculate the average action over a set of causal matrices for a particular temperature.
    
    Parameters:
        causal_matrices (set): A set of unique causal matrices.
        smeared (bool, optional): Whether to apply smearing. Default is True.
        stdim (int, optional): The spacetime dimension. Default is 2.
        epsilon (float, optional): The epsilon parameter for smearing. Default is 0.1.
    """
    save_path = os.path.join(os.path.dirname(__file__), "save_files")
    str_temp = str(Temp).replace(".", "_") 
    
    
    
    try:
        average_action = np.load(os.path.join(save_path, f"average_action_{cardinality}_"+str_temp)+".npy")
        return average_action
    except:
        pass
    causal_bitstrings = causal_matrices.copy()
    causal_matrices = []
    for bitstring in causal_bitstrings:
        matrix = np.zeros((cardinality,cardinality), dtype = np.int32)
        upper_tri_indices = np.triu_indices(cardinality, 1)
        for i in range(len(upper_tri_indices[0])):
            matrix[upper_tri_indices[0][i],upper_tri_indices[1][i]] = bitstring[i]
        causal_matrices.append(matrix)
    
    partition_function = calculate_BD_partition_function(cardinality, causal_matrices, stdim = stdim, epsilon = epsilon, Temp = Temp)
    average_action = 0
    for i, matrix in enumerate(causal_matrices):
        bitstring = causal_bitstrings[i]
        if not is_causal_matrix(matrix):
            raise ValueError("Matrix is not a causal matrix")
        action = calculate_action(matrix, stdim = stdim, epsilon = epsilon)
        mu = calculate_mu(action, partition_function, Temp = Temp)

        average_action += action * mu
    np.save(os.path.join(save_path, f"average_action_{cardinality}_"+str_temp+".npy"), average_action)
    return average_action

def calculate_mu(action, partition_function, Temp):
    return np.exp(-action/Temp)/partition_function
    
def calculate_BD_partition_function(cardinality: int, causal_matrices: set, stdim: int = 2, epsilon: float = 0.1, Temp: float = 1) -> float:
    save_path = os.path.join(os.path.dirname(__file__), "save_files")
    str_temp = str(Temp).replace(".", "_") 
    q = cardinality*(cardinality-1)//2
    try:
        partition_function = np.load(os.path.join(save_path, f"partition_function_{cardinality}_"+str_temp)+".npy")
        return partition_function
    except:   
        partition_function = 0
        for matrix in causal_matrices:
            
            
            action = calculate_action(matrix, stdim = stdim, epsilon = epsilon)
            partition_function += np.exp(-action/Temp)
        
        np.save(os.path.join(save_path, f"partition_function_{cardinality}_"+str_temp+".npy"), partition_function)
        return partition_function




def get_upper_triangular_basis(n: int) -> npt.NDArray:
    """
    
    Generate a basis for the upper triangular part of an n x n matrix.
    Parameters:
        n (int): The cardinality
    
    Returns:
        np.ndarray: The map between the i and j coordinates of the 
            upper triangular part of the matrix and the qubit labelling
    
    """
    
    
    
    q = np.zeros((n,n), dtype = int)
    
    count = int(0)
    for i in range(n):
        for j in range(i+1, n):
            q[i,j] = count
            count += 1
    return q


def num_relations(matrix: npt.NDArray) -> int:
    """
    Calculates the number of relations in a given causal matrix.
    
    Parameters:
    matrix (np.ndarray): The causal matrix to calculate the number of relations of.
    
    Returns:
    int: The number of relations in the causal matrix.
    """
    nr = np.sum(matrix)
    return nr


def height(matrix: npt.NDArray) -> int:
    """
    Calculates the height of a given causal matrix. Ie. the length of the longest chain of relations.
    
    Parameters:
    matrix (np.ndarray): The causal matrix to calculate the height of.
    
    Returns:
    int: The height of the causal matrix.
    """
    n = len(matrix)
    
    longest_path_ending_each = np.zeros(n)
    # for every pair i < j
    for i in range(1, n):
        current_longest_parent = -1
        for j in range(0, i): # search for longest path to a past connection
            
            if matrix[j,i] == 1:
                length_current_parent = longest_path_ending_each[j]
                current_longest_parent = max(current_longest_parent, length_current_parent)
        longest_path_ending_each[i] = current_longest_parent + 1
    
    longest_chain = int(np.max(longest_path_ending_each)) +1
    #+1 as height is the number of nodes, not relations
    return longest_chain

def ordering_fraction(matrix: npt.NDArray) -> float:
    """
    Calculate the ordering fraction of a given matrix.
    The ordering fraction r is the fraction of pairs of elements which are related.
    It is computed as the number of relations (R) divided by the total number of possible pairs (N choose 2).
    
    Parameters:
    matrix (npt.NDArray): A square matrix representing the relations between elements.
    
    Returns:
    float: The ordering fraction of the matrix.
    """
    
    
    
    # The ordering fraction r is the fraction of pairs of elements
    # which are related
    
    #R/(N choose 2) # R is the number of relations
    R = np.sum(matrix)
    n = len(matrix)
    
    return R/((n*(n-1))/2)

def minimal_elements(matrix: npt.NDArray) -> int:
    """
    Counts the number of minimal elements in the causal set.
    A minimal element is defined as an element with no incoming relations,
    i.e., there are no other elements that precede it in the causal set.
    
    Parameters:
    matrix (npt.NDArray): A square adjacency matrix representing the causal set,
        where matrix[i, j] is non-zero if there is a relation from element i to element j.
        
    Returns:
        int: The number of minimal elements in the causal set.
    """
    
    
    
    # Counts the number of minimal elements in the causal set 
    # (elements with no incoming relations)
    
    n = len(matrix)
    minimal_elements = 0
    for i in range(n):
        if np.sum(matrix[:,i]) == 0:
            minimal_elements += 1
    return minimal_elements


def is_critical_pair(x, y, s_mat):
    n = len(s_mat)
    for k in range(y+1, n):
        if s_mat[y,k] ==1: #k is fut(y)
            if s_mat[x,k] != 1: #k is not fut(x)
                # If there is a k such that k is fut(y) but not fut(x), then x and y are not a critical pair
                return False
    for k in range(0, x):
        if s_mat[k,x] ==1:
            if s_mat[k,y] != 1:
                # If there is a k such that k is past(x) but not past(y), then x and y are not a critical pair
                return False
    #print("causal matrix: ", self.causal_matrix)
    return True

    
def is_suitable_pair(x,y, s_mat):
    
    if s_mat[x,y] == 1:
        # If x and y are related, then not a suitable pair
        return False
    
    n= len(s_mat)
    #print("Suitable pair check, C: ", self.causal_matrix)
    for z in range(0, x+1):#z is incpast(x)
        if s_mat[z,x] ==1 or z == x:
            #print("z: ", z)
            for w in range(y, n):#w is incfut(y)
                if s_mat[y,w] ==1 or w == y:
                    #print("w: ", w)
                    if s_mat[z,w] ==1: 
                        #print("Not suitable pair, z, w: ", z,w)
                        # If there is a z in incpast(x) related to w incfut(y), then not suitable
                        return False
            
    return True


def make_basis(n:int) -> npt.NDArray:
    basis = [(j, k) for j in range(n) for k in range(j+1, n)]
    return basis

def is_incpast(a,x,s_mat):
    # If a is in the inclusive past of x, then return True
    # Else, return false
    if s_mat[a,x] == 1 or a == x:
        return True
    else:
        return False

def is_incfut(b,y,s_mat):
    # If a is in the inclusive future of x, then return True
    # Else, return false
    
    if s_mat[y,b] == 1 or b == y:
        return True
    else:
        return False
    
def is_linked(x, y, s_mat):
    linked = False
    if s_mat[x,y] == 1: # If related
        #for all points k
        for k in range(x,y+1): 
            if s_mat[x,k] == 1 and s_mat[k,y] == 1:
                # If there is a k such that k is past(x) and fut(y), then not a link
                return False
        linked = True # If urelated and nothing in between, then linked
    else: # If unrelated
        linked = False
    return linked





def bitstring_to_matrix(bitstring: str) -> npt.NDArray:
    """
    Convert a bitstring representation of an upper triangular matrix into a 2D numpy array.
    
    Parameters:
    bitstring (str): A string of '0's and '1's representing the upper triangular part of the matrix.
    n (int): The size of the resulting square matrix (n x n).
    
    Returns:
    np.ndarray: A 2D numpy array representing the upper triangular matrix.
    """
    n = int((1 + math.sqrt(1 + 8 * len(bitstring))) / 2)  # Solve n(n-1)/2 = len(bitstring)
    
    matrix = np.zeros((n, n), dtype=int)
    upper_tri_indices = np.triu_indices(n, 1)
    
    for i in range(len(upper_tri_indices[0])):
        matrix[upper_tri_indices[0][i], upper_tri_indices[1][i]] = int(bitstring[i])
    
    return matrix

def matrix_to_bitstring(matrix: npt.NDArray) -> str:
    """
    Convert a 2D numpy array representing an upper triangular matrix into a bitstring representation.
    Parameters:
    matrix (np.ndarray): A 2D numpy array representing the upper triangular matrix.
    Returns:
    str: A string of '0's and '1's representing the upper triangular part of the matrix.
    """
    n = matrix.shape[0]
    bitstring = ''
    upper_tri_indices = np.triu_indices(n, 1)
    
    for i in range(len(upper_tri_indices[0])):
        bitstring += str(int(matrix[upper_tri_indices[0][i], upper_tri_indices[1][i]]))
    
    return bitstring



def get_TC_couplings(C):
    """
    get coupling matrix defining the (binary) cost function for the transitive closure constraint for a given cardinality C

    H_TC = \sum_{i<j<k}^N C_ij C_jk (1 - C_ik)
    """
    n = C * (C - 1) // 2
    q = get_upper_triangular_basis(C) 
    
    Q = np.zeros((n, n))
    T = np.zeros((n,n,n))
    
    for i, j, k in combinations(range(C), 3):
        # Identify the indices for the three edges of the triangle
        idx_ij = q[i][j]
        idx_jk = q[j][k]
        idx_ik = q[i][k]
        
        # Quadratic term: + C_ij * C_jk
        # We store in upper-triangular form (a < b)
        a, b = sorted((idx_ij, idx_jk))
        Q[a, b] += 1
        Q[b, a] += 1
        
        # Cubic term: - C_ij * C_jk * C_ik
        # We store in upper-triangular form (a < b < c)
        a, b, c = sorted((idx_ij, idx_jk, idx_ik))
        #T[a, b, c] -= 1
        for perm in permutations([a,b,c]):
            T[perm] -= 1
        
    return Q, T#+Q.T, T+ T.T


def plot_chains_BD(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True):
    avg_energies = []
    for chain in chains:
        #energies = chain.get_current_energy_array()
        energies = [calculate_action(bitstring_to_matrix(state.bitstring), True, stdim=4, epsilon=0.1, first_order_smearing=False, first_order_taylor=False) for state in chain._states_accepted]
        pos = chain.get_pos_array()
        if plot_individual_chains:
            plt.plot(pos, energies, color=color, alpha=0.1)
        avg_energies.append(energies)
    avg_energies = np.array(avg_energies)
    #avg_energy = sum(chain.get_current_energy_array() for chain in chains) / len(chains)
    avg_energy = np.sum(avg_energies, axis=0)/len(chains)
    plt.plot(pos, avg_energy, color=color, label=f"Average {label}")

def plot_chains_height(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True, samp_freq: int = 10):
    avg_heights = []
    for chain in chains:
        heights = [height(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted[0::samp_freq]]
        pos = chain.get_pos_array()[0::samp_freq]
        if plot_individual_chains:
            plt.plot(pos, heights, color=color, alpha=0.1)
        avg_heights.append(heights)
    avg_heights = np.array(avg_heights)
    avg_height = np.sum(avg_heights, axis=0)/len(chains)
    plt.plot(pos, avg_height, color=color, label=f"Average {label}")


def plot_chains_height_alt(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True, samp_freq: int = 10):
    avg_heights = []
    for chain in chains:
        heights = [height(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted[0::samp_freq]]
        pos = chain.get_pos_array()[0::samp_freq]
        if plot_individual_chains:
            plt.plot(pos, heights, color=color, alpha=0.1, marker='o', linewidth = 0)
        avg_heights.append(heights)
    avg_heights = np.array(avg_heights)
    avg_height = np.sum(avg_heights, axis=0)/len(chains)



def plot_chains_mean_height(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True, samp_freq: int = 10, plot_chains = True):
    all_heights = []
    if plot_chains:
        plot_chains_height_alt(chains, color, "chain", plot_individual_chains, samp_freq)

    print("plotting mean height for chains:", label)
    print("of lengths: ", [len(chain._states_accepted) for chain in chains])
    
    for chain in chains:
        heights = [height(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted[0::samp_freq]]
        print("Mean of last 50'%' of states: ", np.mean(np.array(heights)[len(heights)//2:]))
         # only take every samp_freq sample
        pos = chain.get_pos_array()[0::samp_freq]+1
        
        all_heights.append(heights)
        # plot cumulative height average up to each point in the chain
        cumulative_avg_heights = np.cumsum(heights) / (np.arange(len(heights)) + 1)
        if plot_individual_chains:
            plt.plot(pos, cumulative_avg_heights, color=color, alpha=0.1)
            #plt.plot(pos, heights, color=color, alpha=0.1)
        print("Mean of last 50'%' of states: ", np.mean(np.array(heights)[len(heights)//2:]))
    all_heights = np.array(all_heights)
    avg_height = np.sum(all_heights, axis=0)/len(chains)
    cumulative_avg_heights = np.cumsum(avg_height) / (np.arange(len(avg_height)) + 1)
    plt.plot(pos, cumulative_avg_heights, color=color, label=f"Average {label}")
    print()

def plot_chains_ordering_fraction(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True):
    avg_ofs = []
    for chain in chains:
        ofs = [ordering_fraction(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted]
        pos = chain.get_pos_array()
        if plot_individual_chains:
            plt.plot(pos, ofs, color=color, alpha=0.1)
        avg_ofs.append(ofs)
    avg_ofs = np.array(avg_ofs)
    avg_of = np.sum(avg_ofs, axis=0)/len(chains)
    plt.plot(pos, avg_of, color=color, label=f"Average {label}")

def plot_chains_minimal_elements(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True):
    avg_mes = []
    for chain in chains:
        mes = [minimal_elements(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted]
        pos = chain.get_pos_array()
        if plot_individual_chains:
            plt.plot(pos, mes, color=color, alpha=0.1)
        avg_mes.append(mes)
    avg_mes = np.array(avg_mes)
    avg_me = np.sum(avg_mes, axis=0)/len(chains)
    plt.plot(pos, avg_me, color=color, label=f"Average {label}")

def plot_chains_num_relations(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = True):
    avg_nrs = []
    for chain in chains:
        nrs = [num_relations(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted]
        pos = chain.get_pos_array()
        if plot_individual_chains:
            plt.plot(pos, nrs, color=color, alpha=0.1)
        avg_nrs.append(nrs)
    avg_nrs = np.array(avg_nrs)
    avg_nr = np.sum(avg_nrs, axis=0)/len(chains)
    plt.plot(pos, avg_nr, color=color, label=f"Average {label}")
    


def plot_chains_num_relations_hist(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = False, samp_freq: int = 10):
    # change the plotting below, so that it plots the histogram data, but with a scatter plot (instead of bar chart)
    all_nrs = []
    for chain in chains:
        
        # only take every samp_freq sample
        nrs = [num_relations(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted[::samp_freq]]

        all_nrs.append(nrs)

    max_h = max([max(nrs) for nrs in all_nrs])
    for nrs in all_nrs:
        #pos = chain.get_pos_array()[::samp_freq]
        if plot_individual_chains:
            bins, n = np.unique(nrs, return_counts=True)
            N = len(nrs) 
            plt.errorbar(bins, n/N, yerr=np.sqrt(n)/N, fmt='o', color='black')
  


   
        
    all_nrs = np.array(all_nrs)
    
    nrs_flattened = np.array([nrs for nrs in all_nrs]).flatten()
    N = len(nrs_flattened)      
    bins, n = np.unique(nrs_flattened, return_counts=True)
    # Now we find the center of each bin from the bin edges
    plt.errorbar(bins, n/N, yerr=np.sqrt(n)/N, fmt='o', color=color, label=f"{label}")

def plot_chains_height_hist(chains: list[MCMCChain], color: str, label: str, plot_individual_chains: bool = False, samp_freq: int = 10, exact = None):
    # change the plotting below, so that it plots the histogram data, but with a scatter plot (instead of bar chart)
    
    all_nrs = []
    for chain in chains:
        
        # only take every samp_freq sample
        nrs = [height(bitstring_to_matrix(state.bitstring)) for state in chain._states_accepted[::samp_freq]]

        all_nrs.append(nrs)

    max_h = max([max(nrs) for nrs in all_nrs])
    for nrs in all_nrs:
        #pos = chain.get_pos_array()[::samp_freq]
        if plot_individual_chains:
            n, bins = np.histogram(nrs, bins=np.arange(0.5, max_h+1.5), density=True)
            bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]

            N = len(nrs) 
            plt.errorbar(bins_mean, n, yerr=np.sqrt(n)/N, fmt='o', color='black', label=f"Error bars {label}")

        
    all_nrs = np.array(all_nrs)
    
    nrs_flattened = np.array([nrs for nrs in all_nrs]).flatten()
    N = len(nrs_flattened)
    bins, n = np.unique(nrs_flattened, return_counts=True)
    plt.errorbar(bins, n/N, yerr=np.sqrt(n)/N, fmt='o', color=color, label=f"{label}")
    if exact is not None:
        bins_exact, n_exact = exact
        
        plt.errorbar(bins_exact, n_exact, fmt='o', color='k', label=f"Exact")
    #print("number of each height in nrs_flattened: ", np.bincount(nrs_flattened))

def get_BD_couplings_4d(C, epsilon):
    """
    get coupling matrix defining the (binary) cost function for the transitive closure constraint for a given cardinality C

    

    H_BD =  \frac{4}{\sqrt{6}}  \sqrt{\epsilon} *(N-  \sum_{k<m}^N C_{km} (1- 10 \epsilon \sum_{k<l<m}^N C_kl C_lm)
    """
    n = C * (C - 1) // 2
    q = get_upper_triangular_basis(C) 
    
    # Calculate coefficients
    C_base = 4 / math.sqrt(6) * math.sqrt(epsilon)
    C_c = C_base * C
    C_l = C_base *epsilon *-1
    C_t = C_base * epsilon * epsilon * 10

    # Linear terms
    L = np.zeros((n))
    # Cubic terms
    T = np.zeros((n,n,n))
    
    for k in range(C):
        for m in range(k+1, C):
            idx_km = q[k][m]
            L[idx_km] += C_l
            for l in range(k+1, m):
                idx_kl = q[k][l]
                idx_lm = q[l][m]
                #T[idx_kl, idx_lm, idx_km] += C_t
                for perm in permutations([idx_kl, idx_lm, idx_km]):
                    T[perm] += C_t


    # 3. Iterate over all unique triplets (i < j < k)
    # for i, j, k in combinations(range(C), 3):
    #     # Identify the indices for the three edges of the triangle
    #     idx_ij = q[i][j]
    #     idx_jk = q[j][k]
    #     idx_ik = q[i][k]
        
    #     # Quadratic term: + C_ij * C_jk
    #     # We store in upper-triangular form (a < b)
    #     a, b = sorted((idx_ij, idx_jk))
    #     Q[a, b] += 1
        
    #     # Cubic term: - C_ij * C_jk * C_ik
    #     # We store in upper-triangular form (a < b < c)
    #     a, b, c = sorted((idx_ij, idx_jk, idx_ik))
    #     T[a, b, c] -= 1
        
    return C_c, L, T#+ T.T


def generate_kr_ish_matrix(n):
    n2 = n // 2
    n1 = (n - n2) // 2
    n3 = n - n1 - n2
    
    adj = np.zeros((n, n), dtype=int)
    
    l1 = np.arange(0, n1)
    l2 = np.arange(n1, n1 + n2)
    l3 = np.arange(n1 + n2, n)

    # 1. Deterministic 50% density for L1->L2 and L2->L3 using parity
    for i in l1:
        for j in l2:
            u = np.random.rand()
            if u>0.5:
                adj[i, j] = 1
                
    for j in l2:
        for k in l3:
            u = np.random.rand()
            if u>0.5 and (j + k) % 2 == 0:
                adj[j, k] = 1


    adj = transitive_closure(adj)
    # # 2. Enforce Transitivity for L1->L3
    # # A_ik = 1 if there exists j such that A_ij=1 AND A_jk=1
    # forced_l1_l3 = (adj[l1, :][:, l2] @ adj[l2, :][:, l3]) > 0
    
    # # 3. Fill L1->L3 block
    # for i_idx, i in enumerate(l1):
    #     for k_idx, k in enumerate(l3):
    #         if forced_l1_l3[i_idx, k_idx]:
    #             adj[i, k] = 1
    #         else:
    #             # Apply parity to non-forced relations to maintain density
    #             if (i + k) % 2 == 0:
    #                 adj[i, k] = 1
                    
    #return adj

    #return the upper triangle as a bitstring
    return matrix_to_bitstring(adj)

print(generate_kr_ish_matrix(20))