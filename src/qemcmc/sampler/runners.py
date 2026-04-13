import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple
from qemcmc.model import EnergyModel, ConstraintModel
from qemcmc.utils import MCMCChain, MCMCState, get_random_state
from qemcmc.sampler.proposal import Proposal


class Runner:
    """
    Base class for running MCMC routines.
    Subclasses implement specific MCMC based algorithms.
    """

    def __init__(self):
        pass

    def test_probs(self, energy_s: float, energy_sprime: float, temperature: float = 1.0) -> float:
        """
        Calculate the Metropolis acceptance probability.

        This computes exp(-(E(s') - E(s)) / T), used to determine the acceptance
        probability of a new state s' given the current state s.

        Parameters
        ----------
        energy_s : float
            Energy of the current state s.
        energy_sprime : float
            Energy of the proposed state s'.
        temperature : float, optional
            Temperature T, by default 1.0.

        Returns
        -------
        float
            The acceptance probability.
        """
        delta_energy = energy_sprime - energy_s
        if energy_sprime <= energy_s:
            return 1.0
        else:
            exp_factor = np.exp(-delta_energy / temperature)

        return min(1.0, exp_factor)

    def test_accept(self, energy_s: float, energy_sprime: float, temperature: float = 1.0) -> bool:
        """
        Decide whether to accept a proposed state.

        Accepts state s' with probability A = min(1, exp(-(E(s')-E(s))/T)).

        Parameters
        ----------
        energy_s : float
            Energy of the current state s.
        energy_sprime : float
            Energy of the proposed state s'.
        temperature : float, optional
            Temperature T, by default 1.0.

        Returns
        -------
        bool
            True if the new state is accepted, False otherwise.
        """
        acceptance_prob = self.test_probs(energy_s, energy_sprime, temperature)
        return acceptance_prob > np.random.rand()


class MCMCRunner(Runner):
    """
    Orchestrates a standard MCMC run loop.

    This runner uses a given proposal sampler and energy model to generate a Markov chain
    of states. It manages state updates, energy evaluations, and Metropolis acceptance tests.
    The sampler targets the Boltzmann distribution p(s) ∝ exp(-E(s) / T).

    Parameters
    ----------
    model : EnergyModel
        The energy model defining the system.
    temp : float
        The temperature for the Metropolis acceptance criterion.
    """

    def __init__(self, model: EnergyModel, temp: float):
        super().__init__()
        self.model = model
        self.temp = temp

    def run(
        self,
        proposer: Proposal,
        n_hops: int,
        initial_state: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = False,
        sample_frequency: int = 1,
    ) -> MCMCChain:
        """
        Run the MCMC simulation.

        Parameters
        ----------
        proposer : Proposal
            The proposal engine for generating new states.
        n_hops : int
            The number of MCMC steps to perform.
        initial_state : str, optional
            The starting bitstring for the chain. If None, a random state is generated.
        name : str, optional
            A name for the MCMC chain.
        verbose : bool, optional
            If True, enables progress bar and print statements.
        sample_frequency : int, optional
            The frequency at which to sample states for the chain.

        Returns
        -------
        MCMCChain
            The generated Markov chain of states.
        """
        if name is None:
            name = getattr(proposer, "method", "Standard") + " MCMC"

        if initial_state is None:
            initial_state_obj = MCMCState(get_random_state(self.model.n), accepted=True, position=0)
        else:
            initial_state_obj = MCMCState(initial_state, accepted=True, position=0)

        current_state = initial_state_obj
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        if verbose:
            print(f"Starting with: {current_state.bitstring} with energy: {energy_s}")

        mcmc_chain = MCMCChain([current_state], name=name)

        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            s_prime = proposer.update(current_state.bitstring)
            energy_sprime = self.model.get_energy(s_prime)
            accepted = self.test_accept(energy_s, energy_sprime, temperature=self.temp)

            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            # TODO: What is this doing? Why are we ignoring the step at i=0?
            if i % sample_frequency == 0 and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        return mcmc_chain


class ConstrainedMCMCRunner(Runner):
    """
    Orchestrates an MCMC run that enforces a hard constraint.

    If a proposed state does not satisfy the constraint, it is immediately rejected
    without computing its energy or testing the Metropolis criterion.

    Parameters
    ----------
    model : ConstraintModel
        A model that includes a constraint function.
    temp : float
        The temperature for the Metropolis acceptance test.
    reject_invalid : bool, optional
        If True (default), proposed states that violate the constraint are rejected.
    """

    def __init__(self, model: ConstraintModel, temp: float, reject_invalid: bool = True):
        if not isinstance(model, ConstraintModel):
            raise TypeError("Model must be an instance of ConstraintModel.")

        super().__init__()
        self.model = model
        self.temp = temp
        self.constraint_func = self.model.constraint_func
        self.reject_invalid = reject_invalid

    def run(
        self,
        proposer: Proposal,
        n_hops: int,
        initial_state: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = False,
        sample_frequency: int = 1,
        return_rejections: bool = True,
    ) -> Tuple[MCMCChain, int]:
        """
        Run the constrained MCMC simulation.

        Parameters
        ----------
        proposer : Proposal
            The proposal engine for generating new states.
        n_hops : int
            The number of MCMC steps to perform.
        initial_state : str, optional
            The starting bitstring for the chain. Must satisfy the constraint.
            If None, a valid random state is sought.
        name : str, optional
            A name for the MCMC chain.
        verbose : bool, optional
            If True, enables progress bar and print statements.
        sample_frequency : int, optional
            The frequency at which to sample states for the chain.
        return_rejections : bool, optional
            If True, also return the number of rejections due to constraint violations.

        Returns
        -------
        tuple[MCMCChain, int]
            A tuple containing the generated Markov chain and the number of rejections due to constraints. If return_rejections is False, only the MCMCChain is returned.
        """

        if name is None:
            name = getattr(proposer, "method", "Constrained") + " MCMC"

        if initial_state is None:
            if verbose:
                print("No initial state provided, attempting to find a random state that satisfies the constraint...")
            for _ in range(1000):
                candidate = get_random_state(self.model.n)
                if self.constraint_func(candidate):
                    initial_state = candidate
                    break
            if initial_state is None:
                raise ValueError("Could not find a valid initial state. Please provide one.")

        elif not self.constraint_func(initial_state):
            raise ValueError(f"Provided initial state {initial_state} does not satisfy the constraint.")

        initial_state_obj = MCMCState(initial_state, accepted=True, position=0)
        current_state = initial_state_obj
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        mcmc_chain = MCMCChain([current_state], name=name)
        constraint_rejections = 0
        self_rejections = 0
        MH_rejects = 0
        s_prime = None
        pbar = tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose)
        E_diffs = []  # energy difference in proposal
        h_diffs = []  # Hamming distance difference in proposal
        for i in pbar:
            s_prime = proposer.update(current_state.bitstring)
            pbar.set_description(
                f"Run {name} | current state: {current_state.bitstring} | proposing: {s_prime} | avgEdiff: {np.mean(np.abs(E_diffs)):.4f} | avgHdiff: {np.mean(h_diffs):.2f} | constrejecects: {constraint_rejections} | selfrejects: {self_rejections} | MHrejects: {MH_rejects}"
            )
            if s_prime == current_state.bitstring:
                accepted = False
                energy_sprime = energy_s
                self_rejections += 1
            elif self.reject_invalid and not self.constraint_func(s_prime):
                accepted = False
                constraint_rejections += 1
            else:
                energy_sprime = self.model.get_energy(s_prime)
                accepted = self.test_accept(energy_s, energy_sprime, temperature=self.temp)
                E_diffs.append(energy_s - energy_sprime)
                h_diffs.append(sum(c1 != c2 for c1, c2 in zip(current_state.bitstring, s_prime)))
                if not accepted:
                    MH_rejects += 1
            if accepted:
                energy_s = energy_sprime
                current_state = MCMCState(s_prime, accepted, energy_s, position=i)

            # TODO: What is this doing? Why are we ignoring the step at i=0?
            if i % sample_frequency == 0 and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, energy_s, position=i))

        if return_rejections:
            return mcmc_chain, constraint_rejections
        else:
            return mcmc_chain
