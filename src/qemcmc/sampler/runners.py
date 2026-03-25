import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
from qemcmc.utils import MCMCChain, MCMCState, get_random_state

class MCMCRunner:
    """
    Orchestrates the standard MCMC run loop for a given sampler.
    """
    def __init__(self):
        pass

    def run(self, sampler, n_hops: int, initial_state: Optional[str] = None, name: Optional[str] = None, verbose: bool = False, sample_frequency: int = 1):
        if name is None:
            name = getattr(sampler, "method", "Standard") + " MCMC"

        # Either get a random state or use initial state given
        if initial_state is None:
            initial_state_obj = MCMCState(get_random_state(sampler.n_spins), accepted=True, position=0)
        else:
            initial_state_obj = MCMCState(initial_state, accepted=True, position=0)

        # set initial state
        current_state = initial_state_obj
        energy_s = sampler.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        if verbose:
            print("starting with: ", current_state.bitstring, "with energy:", energy_s)

        # define MCMC chain
        mcmc_chain = MCMCChain([current_state], name=name)

        # run MCMC
        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            # Propose a new state via the decoupled sampler
            s_prime = sampler.update(current_state.bitstring)

            # Find energy of the new state
            energy_sprime = sampler.model.get_energy(s_prime)

            # Decide whether to accept the new state
            accepted = sampler.test_accept(energy_s, energy_sprime, temperature=sampler.temp)

            # If accepted, update current_state
            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            # if time to sample, add state to chain
            if i // sample_frequency == i / sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        return mcmc_chain


class ConstrainedMCMCRunner:
    """
    Orchestrates an MCMC run loop but enforces a hard constraint on the proposed states.
    If a proposed state does not satisfy the constraint, it is immediately rejected
    without computing its energy or testing the Metropolis criteria.
    """
    def __init__(self, constraint_func: Callable[[str], bool]):
        """
        Args:
            constraint_func (Callable): A function that takes a bitstring (str) 
                                        and returns True if valid, False otherwise.
        """
        self.constraint_func = constraint_func

    def run(self, sampler, n_hops: int, initial_state: Optional[str] = None, name: Optional[str] = None, verbose: bool = False, sample_frequency: int = 1):
        if name is None:
            name = getattr(sampler, "method", "Constrained") + " MCMC"

        if initial_state is None:
            # Attempt to find a random initial state that meets the constraint
            for _ in range(1000):
                candidate = get_random_state(sampler.n_spins)
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
        energy_s = sampler.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        mcmc_chain = MCMCChain([current_state], name=name)

        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            s_prime = sampler.update(current_state.bitstring)

            # 1. Constraint Check FIRST
            if not self.constraint_func(s_prime):
                accepted = False # Instant rejection, no energy calculations performed
            else:
                # 2. Standard Metropolis-Hastings Check
                energy_sprime = sampler.model.get_energy(s_prime)
                accepted = sampler.test_accept(energy_s, energy_sprime, temperature=sampler.temp)

            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            if i // sample_frequency == i / sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        return mcmc_chain
    
