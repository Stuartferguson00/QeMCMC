import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Callable
from qemcmc.model.energy_model import EnergyModel, ConstraintModel
from qemcmc.utils import MCMCChain, MCMCState, get_random_state


class Runner:
    """
    Base class for running MCMC routines. 
    Subclasses implement specific MCMC based algorithms .
    """
    def __init__(self):
        pass
    
    def test_probs(self, energy_s: float, energy_sprime: float, temperature: float = 1.0) -> float:
        """
        Calculate the probability ratio between two states based on their energies.
        This function computes the exponential factor used in the Metropolis-Hastings
        algorithm to determine the acceptance probability of a new state s' given
        the current state s. The probability ratio is calculated as exp(-(E(s') - E(s)) / T),
        where E(s) and E(s') are the energies of the current and proposed states, respectively,
        and T is the temperature.
        Args:
            energy_s (float): The energy of the current state s.
            energy_sprime (float): The energy of the proposed state s'.
        Returns:
            float: The probability ratio exp(-(E(s') - E(s)) / T).
        """

        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        if energy_sprime < energy_s:
            exp_factor = 1
        else:
            exp_factor = np.exp(-delta_energy / temperature)

        acceptance = min(1, exp_factor)
        return acceptance

    def test_accept(self, energy_s: float, energy_sprime: float, temperature: float = 1.0) -> MCMCState:
        """
        Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
        and s_init with probability 1-A.
        """
        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        # with warnings.catch_warnings():
        #    warnings.simplefilter("error", RuntimeWarning)
        try:
            exp_factor = np.exp(-delta_energy / temperature)
        except RuntimeWarning:
            if energy_sprime < energy_s:
                exp_factor = 1
            else:
                exp_factor = 0

            # print("Error in exponantial: delta_energy = ", delta_energy, "temperature = ", temperature, " energy_s = ", energy_s, " energy_sprime = ", energy_sprime)

        acceptance = min(1, exp_factor)  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

        return acceptance > np.random.rand()


class MCMCRunner(Runner):
    
    def __init__(self, model:EnergyModel, temp:float):
        """
        Orchestrates the standard MCMC run loop for a given Proposal sampler and EnergyModel. It manages the state updates, energy evaluations, and Metropolis acceptance tests, while recording the Markov chain of states.
        The sampler targets the Boltzmann distribution

            p(s) ∝ exp(-E(s) / T)

        where ``E(s)`` is the energy of configuration ``s`` provided by the energy model.
        """
        
        super().__init__()
        self.model = model
        self.temp = temp

    

    def run(self, proposer, n_hops: int, initial_state: Optional[str] = None, name: Optional[str] = None, verbose: bool = False, sample_frequency: int = 1):
        if name is None:
            name = getattr(proposer, "method", "Standard") + " MCMC"

        # Either get a random state or use initial state given
        if initial_state is None:
            initial_state_obj = MCMCState(get_random_state(self.model.n_spins), accepted=True, position=0)
        else:
            initial_state_obj = MCMCState(initial_state, accepted=True, position=0)

        # set initial state
        current_state = initial_state_obj
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        if verbose:
            print("starting with: ", current_state.bitstring, "with energy:", energy_s)

        # define MCMC chain
        mcmc_chain = MCMCChain([current_state], name=name)

        # run MCMC
        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            # Propose a new state via the decoupled sampler
            s_prime = proposer.update(current_state.bitstring)

            # Find energy of the new state
            energy_sprime = self.model.get_energy(s_prime)

            # Decide whether to accept the new state
            accepted = self.test_accept(energy_s, energy_sprime, temperature=self.temp)

            # If accepted, update current_state
            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            # if time to sample, add state to chain
            if i // sample_frequency == i / sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        return mcmc_chain


class ConstrainedMCMCRunner(Runner):

    def __init__(self, model:ConstraintModel, temp:float, reject_invalid: bool = True):
        """
        Orchestrates an MCMC run loop but enforces a hard constraint on the proposed states.
        If a proposed state does not satisfy the constraint, it is immediately rejected
        without computing its energy or testing the Metropolis criteria.
        Args:
            model (ConstraintModel): An model that includes a constraint function.
            temp (float): The temperature for the Metropolis acceptance test.
            reject_invalid (bool): If True, proposed states that do not satisfy the constraint are rejected. 
                If False, they are accepted. Defaults to True.
        """
        if not isinstance(model, ConstraintModel):
            if isinstance(model, EnergyModel):
                raise TypeError("Model must be an instance of ConstraintModel, not EnergyModel.")

        super().__init__()
        self.model = model
        self.temp = temp
        self.constraint_func = self.model.constraint_func
        self.reject_invalid = reject_invalid


        


    def run(self, proposer, n_hops: int, initial_state: Optional[str] = None, name: Optional[str] = None, verbose: bool = False, sample_frequency: int = 1):
        if name is None:
            name = getattr(proposer, "method", "Constrained") + " MCMC"

        if initial_state is None:
            if verbose:
                print("no initial state provided, attempting to find a random state that satisfies the constraint...")
            # Attempt to find a random initial state that meets the constraint
            for _ in range(1000):
                candidate = get_random_state(self.model.n_spins)
                if self.constraint_func(candidate):
                    initial_state = candidate
                    break
            if initial_state is None:
                raise ValueError("Could not find a valid initial state satisfying the constraint. Please provide one manually.")
        
        else:
            if not self.constraint_func(initial_state):
                raise ValueError(f"Provided initial state {initial_state} does not satisfy the constraint.")

        initial_state_obj = MCMCState(initial_state, accepted=True, position=0)
        current_state = initial_state_obj


        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        mcmc_chain = MCMCChain([current_state], name=name)
        constraint_rejections = 0
        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            s_prime = proposer.update(current_state.bitstring)

            # 1. Constraint Check FIRST
            if self.reject_invalid and not self.constraint_func(s_prime):
                accepted = False # Instant rejection, no energy calculations performed
                constraint_rejections += 1
            else:
                # 2. Standard Metropolis-Hastings Check
                energy_sprime = self.model.get_energy(s_prime)
                accepted = self.test_accept(energy_s, energy_sprime, temperature=self.temp)

            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            if i // sample_frequency == i / sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        return mcmc_chain, constraint_rejections
    
