from qemcmc.model import EnergyModel
from qemcmc.sampler import Proposal
from qemcmc.utils import get_random_state
import numpy as np
from CSThelpers import *

class CSTClassicalProposal(Proposal):
    """
    Classical Markov Chain Monte Carlo proposer for Causal Set Theory sampling.

    This class implements purely classical proposal mechanisms for MCMC.
    New candidate states are generated either bythe link move, the relation move or by a mixture. See "Onset of the Asymptotic Regime for Finite Orders" https://arxiv.org/abs/1504.05902 for more details on these moves.

    Parameters
    ----------
    model : EnergyModel
        Energy model defining the target Boltzmann distribution.
    method : str, optional
        Proposal mechanism used to generate candidate states.

        - ``"link"`` : propose a completely random link configuration.
        - ``"relation"`` : flip a single randomly chosen relation.
        - ''both'' : randomly choose between 'link' and 'relation' for each proposal.

        Default is ``"link"``.
    """

    def __init__(self, model: EnergyModel, method: str = "link"):
        super().__init__(model)
        self.method = method

        

    def update(self, s) -> str:
        if self.method == "link":
            return self.link_move(s)
        elif self.method == "relation":
            return self.relation_move(s)
        elif self.method == "both":
            return self.update_both(s)
        else:
            raise ValueError(f"Method '{self.method}' is not supported. Choose from 'link', 'relation', or 'both'.")

    

    def link_move(self, s) -> str:
        """Perform a link move operation on a binary string `s` representing the upper 
        triangular part of a causal matrix. The function modifies the causal matrix 
        by either removing or adding a link between two randomly selected nodes, while
        ensuring that the resulting matrix maintains transitive closure. As described in Henson paper
        
        Args:
            s (str): A binary string representing the upper triangular part of the 
                causal matrix.
        Returns:
            str: A binary string representing the updated upper triangular part of 
                the causal matrix after the link move operation.
        
        """
        
        C =  int((1 + math.sqrt(1 + 8 * self.n_spins)) / 2)
        s_mat = np.zeros((C, C), dtype=np.int32)
        s_mat[np.triu_indices(C, 1)] = [int(bit) for bit in s]
        
        # pick two random elements i and j
        i = np.random.randint(0, C)
        j = np.random.randint(0, C)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, C)
            
        y = max(i,j)
        x = min(i,j)
        
        
        if is_linked(x,y, s_mat):

            
            """
            # Find all pairs k, l between incpast(x) and incfut(y)
            # Remove relations
            
            self.causal_matrix[x,y] = 0 # Unrelate x and y
            
            for k in range(0, x+1):
                if self.causal_matrix[k,x] == 1: # If k is incpast(x):
                    for l in range(y, self.n):
                        if self.causal_matrix[y,l] == 1: # If l is incfut(y):   
                            self.causal_matrix[k,l] = 0 # Unrelate all k (incpast(x)) and l (incfut(y))
            
            # restore every element by transitivity (where relations are inferred by elements other than x and y)
            # Think that since it is incfut and incpast, 
            # we can just do full transitive closure, although this is innefficient
            self.causal_matrix = self.transitive_closure(self.causal_matrix)
            
            """
            
            link_matrix = transitive_reduction(s_mat)
            link_matrix[x,y] = 0
            s_mat = transitive_closure(link_matrix)

            
        elif is_suitable_pair(x,y, s_mat):
            #print("suitable pair")

            
            #self.causal_matrix[x,y] = 1 # Relate x and y
            
            link_matrix = transitive_reduction(s_mat)
            
            if link_matrix[x,y] != 0:
                print("suitable link error")   

            link_matrix[x,y] = 1
            
            
            s_mat = transitive_closure(link_matrix)
            
            
            
            link_matrix_new = transitive_reduction(s_mat)
            
            if link_matrix_new[x,y] != 1:
                print("link move no change error")
                print("x,y: ", x,y)
                print(link_matrix)
                print(link_matrix_new)            
            
            
            """
            Doing exactly as in the paper doesnt work as it doesnt escape anti-chains
            original_hamm = np.sum(self.causal_matrix)
            
            # add relation between all incpast(x) and incfut(y)
            for k in range(0, x+1):
                if self.causal_matrix[k,x] == 1: # If k is incpast(x):
                    
                    for l in range(y, self.n):
                        if self.causal_matrix[y,l] == 1: # If l is incfut(y): 
                            
                            self.causal_matrix[k,l] = 1 # Relate all k (incpast(x)) and l (incfut(y))
            end_hamm = np.sum(self.causal_matrix)
            
            print("hamming weight change: ", end_hamm - original_hamm)"""
            
        else:
            #print("not suitable or linked pair")
            pass
        #return s_mat
        return "".join(str(bit) for bit in s_mat[np.triu_indices(C, 1)])
            
    def relation_move(self, s)-> str:
        """
        Perform a relation move on a binary string representation of an upper triangular matrix.
        This function modifies the binary string `s` by randomly selecting two indices (i, j),
        ensuring they are distinct, and then determining whether to update the corresponding
        entry in the upper triangular matrix based on specific conditions.
        Args:
            s (str): A binary string representing the upper triangular part of an n x n matrix,
                excluding the diagonal. The length of the string should be `n * (n - 1) / 2`.
        Returns:
            str: A modified binary string representing the updated upper triangular matrix.
        
        """
        

        C =  int((1 + math.sqrt(1 + 8 * self.n_spins)) / 2)
        s_mat = np.zeros((C,C), dtype=np.int32)
        s_mat[np.triu_indices(C, 1)] = [int(bit) for bit in s]
        # pick two random elements i and j
        i = np.random.randint(0, C)
        j = np.random.randint(0, C)
        
        # make sure i != j
        while i == j:
            j = np.random.randint(0, C)
            
        y = max(i,j)
        x = min(i,j)
        
        
        
        if is_linked(x,y, s_mat):
            #print("linked")
            s_mat[x,y] = 0
        elif is_critical_pair(x,y,s_mat):
            #print("critical pair")
            s_mat[x,y] = 1
        else:
            #print("not critical or linked pair")
            pass
        return "".join(str(bit) for bit in s_mat[np.triu_indices(C, 1)])
    
    def update_both(self, s):
        if np.random.rand() < 0.5:
            return self.link_move(s)
        else:
            return self.relation_move(s)
