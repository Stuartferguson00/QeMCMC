import numpy as np

from qemcmc.model.energy_model import EnergyModel
from qemcmc.sampler import Proposal
from tqdm import tqdm
import scipy as sp
from qemcmc.sampler.runners import Runner

class SpectralGap(Runner):
    """
    Class that finds the spectral gap, and the acceptance and proposal matrices for a given mcmc.
    """

    def __init__(self, proposal: Proposal, model:EnergyModel, temp: float = 1.0):
        self.proposal = proposal
        self.model = model
        self.temp = temp

    def find_acceptance_matrix(self):
        """
        Function to find the acceptance matrix for a given model instance.

        Returns:
            A (np.ndarray): The acceptance matrix for the mcmc

        """

        num_states = 2**self.proposal.n_spins

        A = np.zeros((num_states, num_states))

        energies = self.model.get_all_energies()

        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    A[i][j] = self.test_probs(energies[i], energies[j], temperature=self.temp)
                else:
                    A[i][j] = 0

        return A

    def find_prob_matrix_local(self):
        """
        Function to find the proposal matrix for a given local chain.
        Returns:
            Q (np.ndarray): The Q matrix for local proposal
        """

        possible_states = self.model.S
        # TODO: define S separately, not inside the model since S is the state space and will be pretty inefficient otherwise

        Q = np.zeros((2**self.model.n_spins, 2**self.model.n_spins))

        # loop throguh and find the difference in bitstrings.
        # When the ith bitstring is different (by a valua of 1) from the jth bitstring add 1 to Q[i,j]
        for i in range(2**self.model.n_spins):
            for j in range(2**self.model.n_spins):
                sm = 0
                for k in range(self.model.n_spins):
                    sm += abs(int(possible_states[i][k]) - int(possible_states[j][k]))

                # ie if the number of different strings is the size of the cluster (= 1 for local)
                if sm == 1:
                    Q[i, j] = 1

        row_sums = Q.sum(axis=1)
        Q = Q / row_sums[:, np.newaxis]

        return Q

    def find_prob_matrix_uniform(self):
        """
        Function to find the proposal matrix for a given uniform chain.

        Returns:
            Q (np.ndarray): The Q matrix for uniform proposal
        """

        Q = np.ones((2**self.model.n_spins, 2**self.model.n_spins)) / (self.proposal.n_spins**2 - 1)
        row_sums = Q.sum(axis=1)
        Q = Q / row_sums[:, np.newaxis]

        return Q

    def find_prob_matrix_quantum(self):
        """
        Function to find the proposal matrix for a given QeMCMCChain object.

        Returns:
            Q (np.ndarray): The Q matrix for quantum proposal
        """
        if not isinstance(self.proposal.gamma, (int, float)):
            raise ValueError("gamma must be a float to find the proposal matrix for quantum proposal, not a tuple. Got: ", self.proposal.gamma, "of type: ", type(self.proposal.gamma))
        if not isinstance(self.proposal.time, int):
            raise ValueError("time must be an integer to find the proposal matrix for quantum proposal, not a tuple. Got: ", self.proposal.time, "of type: ", type(self.proposal.time) )
        Q = np.zeros((2**self.model.n_spins, 2**self.model.n_spins))

        for i in range(2**self.model.n_spins):
            Q[i, :] += abs(self.proposal.CM.get_state_vector(self.model.S[i], self.proposal.gamma, self.proposal.time, self.proposal.coupling_weights))**2
            #get_output_statevector(self.proposal.model.S[i])
        Q = Q

        return Q

    def find_proposal_matrix_brute_force(self, multiple=100):
        num_states = 2**self.model.n_spins
        possible_states = self.model.S

        Q = np.zeros((num_states, num_states))

        for i, s in tqdm(enumerate(possible_states), total=num_states, desc="Processing Q brute force"):
            for _ in range(multiple):
                s_prime = self.proposal.update(s)
                j = int(s_prime, 2)
                Q[i][j] += 1
        Q = Q / multiple

        return Q

    def find_proposal_matrix(self):
        """
        Function to find the proposal matrix for a given mcmc.
        This is not done by brute force
        """

        if self.proposal.method == "local":
            Q = self.find_prob_matrix_local()
        elif self.proposal.method == "uniform":
            Q = self.find_prob_matrix_uniform()
        elif self.proposal.method == "quantum":
            Q = self.find_prob_matrix_quantum()
        else:
            raise ValueError("Method not recognised. Only 'local', 'uniform' or 'quantum' proposal methods are implimented in find_proposal_method.")

        return Q

    def find_spec_gap(self, A=None, Q=None):
        """

        Function to find the spectral gap of a given mcmc.
        Args:
            A (np.ndarray): The acceptance matrix for the mcmc (optional, if not given, will be calculated)
            Q (np.ndarray): The proposal matrix for the mcmc (optional, if not given, will be calculated)

        """

        if A is None:
            A = self.find_acceptance_matrix()
        if Q is None:
            Q = self.find_proposal_matrix()
            # Q = self.find_proposal_matrix_brute_force(multiple = 10*2**self.mcmc.n_spins)

        P = np.multiply(Q, A)

        # account for rejected swaps to add to s = s' matrix element
        for i in range(P.shape[0]):
            s = np.sum(P[i, :]) - P[i, i]
            P[i, i] = 1 - s

        # find eigenvalues
        e_vals, e_vecs = sp.linalg.eig(P)
        e_vals = np.flip(np.sort(abs(e_vals)))

        # find spectral gap
        delta = e_vals[1]
        delta = 1 - delta

        return delta
