from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from qemcmc.model.constraint_model import ConstraintModel
from qemcmc.model.energy_model import EnergyModel
from .basecircuitmaker import BaseCircuitMaker

class AdiabaticCircuitMaker(BaseCircuitMaker):
    """
        The total Hamiltonian simulated by the circuit is
    
            ``H = H_mixer + γ α H_problem``
    
        where ``H_problem`` encodes the classical energy model and ``H_mixer``
        corresponds to a transverse-field term. The evolution time ``t`` and
        Trotter step ``δt`` is given, resulting in ``r = t/δt`` steps. The evolution is implemented
        via ``qml.ApproxTimeEvolution``.
    """

    
    
    def __init__(self, model: EnergyModel):#, kappa:float | List[float] | Tuple[float, float] = 1.0, dt_overide: float | None = None):
        super().__init__(model)

        # self.kappa = kappa
        # self.dt_overide = dt_overide

        self.method = "adiabatic_quantum"

    

    
    def get_sample(self, s_cg: str, alpha: float, kappa: float, mix_weight: float, local_couplings: list, weights: list[float] = None, dt: float = 0.3) -> str:
        """
        Generate a single sample by evolving the system under a time dependent hamiltonian and measuring.

        Parameters
        ----------
        s_cg : str
            Input bitstring for the subgroup.
        alpha : float
            Time for adiabatic ramp
        kappa : float
            Time for plateaux
        mix_weight : float
            Coefficient for the mixer Hamiltonian.
        local_couplings : list
            Coupling tensors for the subgroup.
        weights : list of float, optional
            Coefficients for the problem Hamiltonian terms. Defaults to ones.
        dt : float
            Timestep size for second order trotter

        Returns
        -------
        str
            A single bitstring sample.
        """
        if weights is None:
            weights = np.ones(len(local_couplings))

        num_wires = len(s_cg)
        dev = self._get_device(num_wires)

        coeff_mixer = mix_weight
        coeff_problem = weights

        
        if len(local_couplings) == 0:
            # If there are no local couplings, just return the input state
            print("no non-zero local couplings, skipping evolution and returning input state")
            return s_cg

        H_m = self.get_mixer_hamiltonian(num_wires)
        H_p = [self.get_problem_hamiltonian(couplings=[local_couplings[i]], sign=coeff_problem[i]) for i in range(len(local_couplings))]


        r = max(1, int(round((alpha * 2 + kappa) / dt)))

        @qml.qnode(dev, shots=1)
        def quantum_evolution(input_string):
            # stat eprep
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)

            # manually slice up evolution
            for step in range(r):
                # choose midpoint
                t_mid = (step + 0.5) * dt

                # Find instantaneous Hmailtonian
                h = self.linear_schedule(t_mid, alpha, kappa)
                H_step = qml.Hamiltonian([h*coeff_mixer] + list(np.ones(len(local_couplings))), [H_m] + H_p)

                # compute step
                qml.ApproxTimeEvolution(H_step, dt, 2)

            return qml.sample()
        
        sample = quantum_evolution(s_cg)[0]
        bitstring = "".join(str(int(b)) for b in sample)
        return bitstring


    def linear_schedule(self, t: float, alpha: float, kappa: float) -> float:
        """Calculates linear ramp weights with a plateau of duration self.kappa and ramp duration alpha.

        Returns:
            h: 0 -> 1 -> 1 -> 0
            as
            t: 0 -> alpha -> alpha + kappa -> 2* alpha + kappa
        """
        total_time = alpha * 2
        
        if total_time == 0:
            return 1.0
        elif t <= 0:
            return 0.0
        elif t <= alpha:
            return t / alpha
        elif t <= alpha + kappa:
            return 1.0
        elif t < total_time + kappa:
            return (total_time + kappa - t) / alpha
        else:
            return 0.0


    def get_state_vector(self, s_cg: str, alpha: float, kappa: float, mix_weight: float, weights: list[float] = None, dt: float = 0.3) -> str:
        """
        Generate a single sample by evolving the system under a time dependent hamiltonian and measuring.

        Parameters
        ----------
        s_cg : str
            Input bitstring for the subgroup.
        alpha : float
            Time for adiabatic ramp
        kappa : float
            Time for plateaux
        mix_weight : float
            Coefficient for the mixer Hamiltonian.
        
        weights : list of float, optional
            Coefficients for the problem Hamiltonian terms. Defaults to ones.
        dt : float
            Timestep size for second order trotter

        Returns
        -------
        str
            A single bitstring sample.
        """
        local_couplings = self.model.normalised_couplings

        if weights is None:
            weights = np.ones(len(local_couplings))

        num_wires = len(s_cg)
        dev = self._get_device(num_wires)

        coeff_mixer = mix_weight
        coeff_problem = weights

        
        if len(local_couplings) == 0:
            # If there are no local couplings, just return the input state
            print("no non-zero local couplings, skipping evolution and returning input state")
            return s_cg

        H_m = self.get_mixer_hamiltonian(num_wires)
        H_p = [self.get_problem_hamiltonian(couplings=[local_couplings[i]], sign=coeff_problem[i]) for i in range(len(local_couplings))]



        r = max(1, int(round((alpha * 2 + kappa) / dt)))

        @qml.qnode(dev)
        def quantum_evolution(input_string):
            # stat eprep
            for i, bit in enumerate(input_string):
                if bit == "1":
                    qml.PauliX(i)

            # manually slice up evolution
            for step in range(r):
                # choose midpoint
                t_mid = (step + 0.5) * dt

                # Find instantaneous Hmailtonian
                h = self.linear_schedule(t_mid, alpha, kappa)
                H_step = qml.Hamiltonian([h*coeff_mixer] + list(np.ones(len(local_couplings))), [H_m] + H_p)

                # compute step
                qml.ApproxTimeEvolution(H_step, dt, 2)

            return qml.state()
        
        state_vector = quantum_evolution(s_cg)
        return state_vector
        


    # def get_state_vector(self, s_cg: str, weights: List[float], time: float, r: int, mix_weight: float) -> np.ndarray:
    #     """
        
    #     """
    #     raise NotImplementedError
    #     local_couplings = self.model.normalised_couplings
    #     if weights is None:
    #         weights = np.ones(len(local_couplings))
    #     num_wires = len(s_cg)
    #     dev = self._get_device(num_wires)

        

    #     coeff_mixer = mix_weight
    #     coeff_problem = weights

    #     # coeff_problem = [coeff for i, coeff in enumerate(coeff_problem) if local_couplings[i].ndim > 0 and np.any(local_couplings[i] != 0)]
    #     # local_couplings = [coupling for coupling in local_couplings if coupling.ndim > 0 and np.any(coupling != 0)]
    #     if len(local_couplings) == 0:
    #         # If there are no local couplings, just return the input state
    #         print("no non-zero local couplings, skipping evolution and returning input state")
    #         return s_cg

    #     # H_total = qml.Hamiltonian(
    #     #     [coeff_mixer] + list(np.ones(len(local_couplings))),
    #     #     [self.get_mixer_hamiltonian(num_wires)] + [self.get_problem_hamiltonian(couplings=[local_couplings[i]], sign=coeff_problem[i]) for i in range(len(local_couplings))],
    #     # )
    #     H_m = self.get_mixer_hamiltonian(num_wires)
    #     H_p = [self.get_problem_hamiltonian(couplings=[local_couplings[i]], sign=coeff_problem[i]) for i in range(len(local_couplings))]
    #     if isinstance(self.kappa, (list, tuple)):
    #         # choose kappa from range
    #         kappa = float(np.random.uniform(self.kappa[0], self.kappa[1]))
    #     elif isinstance(self.kappa, (int, float)):
    #         kappa = float(self.kappa)
    #     else:
    #         kappa = float(self.kappa)

    #     if self.dt_overide is not None:
    #         dt = self.dt_overide
    #         r = max(1, int(round((time + kappa) / dt)))

    #     # set qnode to use our device with dynamically chosen wires
    #     @qml.qnode(dev)
    #     def quantum_evolution(input_string):
    #         for i, bit in enumerate(input_string):
    #             if bit == "1":
    #                 qml.PauliX(i)
    #         dt = (time + kappa) / r
    #         for step in range(r):
                
    #             t_mid = (step + 0.5) * dt
    #             # e.g. Linear adiabatic ramp
    #             h = self.schedule_factor(t_mid, time, kappa)
                
    #             H_step = qml.Hamiltonian([h*coeff_mixer] + list(np.ones(len(local_couplings))), [H_m] + H_p)
                
    #             # Single Trotter slice of duration dt
    #             qml.ApproxTimeEvolution(H_step, dt, 1)

    #         return qml.state()
            
    #     state_vector = quantum_evolution(s_cg)
    #     return state_vector
        
    

    def update(self, s: str, subgroup_choice: list[int], local_couplings: list, gamma: float, alpha: float, kappa: float, dt:float) -> str:
        """
        Update a bitstring by evolving a subgroup.

        This performs a time evolution on a coarse-grained Hamiltonian to get s' from s.

        Parameters
        ----------
        s : str
            The initial bitstring.
        subgroup_choice : list of int
            Indices of the subgroup to evolve.
        local_couplings : list
            Coupling tensors for the subgroup.
        gamma : float
            Mixing parameter.
        alpha : float
            Adiabatic ramp time.
        kappa: float
            Plateaux time
        dt : float
            Trotter step size for the approximate time evolution.

        Returns
        -------
        str
            The updated bitstring s'.
        """

        self._validate_bitstring(s)
        # Get s_cg' for the subgroup and reconstruct full s' using s and s_cg'
        s_cg = "".join([s[i] for i in subgroup_choice])
        s_cg_prime = self.get_sample(s_cg, alpha, kappa, gamma, local_couplings, dt = dt)
        s_list = list(s)
        for i, global_index in enumerate(subgroup_choice):
            s_list[global_index] = s_cg_prime[i]
        return "".join(s_list)

