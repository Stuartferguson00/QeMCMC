import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy import sparse
import itertools
import matplotlib.pyplot as plt

# Internal imports from our QeMCMC package
from qemcmc.coarse_grain import CoarseGraining
from qemcmc.sampler import ClassicalProposal, QeProposal
from CST_classical_proposals import CSTClassicalProposal
from qemcmc.sampler.runners import MCMCRunner, ConstrainedMCMCRunner
from qemcmc.utils import plot_chains, get_random_state
from qemcmc.model import EnergyModel, ConstraintModel, ModelMaker, constraint_model
from CST_helpers import *
from tabulate import tabulate

C = 10# Cardinality
n = C * (C - 1) // 2 # Number of (Qu)bits needed to represent the spacetime
temp = 0.01
epsilon = 0.1

print(f"Using {n} bits to represent a causal set of cardinality {C}.")
# calculate_action
BD_couplings = get_BD_couplings_4d(C, epsilon=epsilon)
BD_couplings_list = [BD_couplings[0], BD_couplings[1], BD_couplings[2]]

# Make baseline Ising gmodel
BD_action_model = EnergyModel(n, couplings = BD_couplings_list, name = "BD Action Model", cost_function_signs =[1,1,1], model_type = "binary")



def constraint_checker_func(bitstring: str) -> bool:
    """In {0, 1} representation, this constraint means no two adjacent bits can both be 1"""
    return is_causal_matrix(bitstring_to_matrix(bitstring))


TC_couplings = get_TC_couplings(C)
TC_couplings_list = [TC_couplings[0], TC_couplings[1]]
TC_model = ConstraintModel(n, constraint_couplings = [coupling for coupling in TC_couplings_list], name="TC Constraint Model", constraint_func=constraint_checker_func, constraint_signs = [1 for coupling in TC_couplings_list], couplings = None, model_type ="binary", cost_function_signs=None)


initial_state_0 = "0"*n
initial_state_1 = "1"*n


# constraint_model = ConstraintModel(n, constraint_couplings = [J_constraint,], couplings = base_model.couplings, name="Constraint Model", cost_function_signs=base_model.cost_function_signs, constraint_func=constraint_checker_func, constraint_signs = [-1,])
BD_TC_model = ConstraintModel(n, constraint_couplings = [coupling for coupling in TC_couplings_list], name="TC Constraint Model", constraint_func=constraint_checker_func, constraint_signs = [1 for coupling in TC_couplings_list], couplings = [coupling for coupling in BD_couplings_list], model_type ="binary", cost_function_signs=[1 for coupling in BD_couplings_list])
Runner = ConstrainedMCMCRunner(BD_TC_model, temp=temp, reject_invalid=True)


num_steps = 1000

cst_proposal_both = CSTClassicalProposal(TC_model, method="both")
cst_chain_both_0, cst_rejections_both_0 = Runner.run(cst_proposal_both, n_hops=num_steps, initial_state=initial_state_0, verbose = True)
cst_chain_both_1, cst_rejections_both_1 = Runner.run(cst_proposal_both, n_hops=num_steps, initial_state=initial_state_1, verbose = True)

cst_proposal_link = CSTClassicalProposal(TC_model, method="link")
cst_chain_link_0, cst_rejections_link_0 = Runner.run(cst_proposal_link, n_hops=num_steps, initial_state=initial_state_0, verbose = True)
cst_chain_link_1, cst_rejections_link_1 = Runner.run(cst_proposal_link, n_hops=num_steps, initial_state=initial_state_1, verbose = True)

cst_proposal_relation = CSTClassicalProposal(TC_model, method="relation")
cst_chain_relation_0, cst_rejections_relation_0 = Runner.run(cst_proposal_relation, n_hops=num_steps, initial_state=initial_state_0, verbose = True)
cst_chain_relation_1, cst_rejections_relation_1 = Runner.run(cst_proposal_relation, n_hops=num_steps, initial_state=initial_state_1, verbose = True)



loc_proposal = ClassicalProposal(TC_model, method="local")
loc_chain_0, loc_rejections_0 = Runner.run(loc_proposal, n_hops=num_steps, initial_state=initial_state_0, verbose = True)
loc_chain_1, loc_rejections_1 = Runner.run(loc_proposal, n_hops=num_steps, initial_state=initial_state_1, verbose = True)

uniform_proposal = ClassicalProposal(TC_model, method="uniform")
uniform_chain_0, uni_rejections_0 = Runner.run(uniform_proposal, n_hops=num_steps, initial_state=initial_state_0, verbose = True)
uniform_chain_1, uni_rejections_1 = Runner.run(uniform_proposal, n_hops=num_steps, initial_state=initial_state_1, verbose = True)



num_steps_q = 2
def run_qe(init_state):
    cg = CoarseGraining(n, repeated=False)
    qe_proposal = QeProposal(TC_model, gamma=0.05, time=(1, 40), m=4)
    return Runner.run(qe_proposal, n_hops=num_steps_q, initial_state=init_state, verbose=True)

print("\n--- Running Quantum Chains in Parallel ---")
qe_results = Parallel(n_jobs=2)(
    delayed(run_qe)(s) for s in [initial_state_0, initial_state_1]
)

(qe_chain_0, qe_re_0), (qe_chain_1, qe_re_1) = qe_results





plt.title("BD action vs steps uniform sampling")
plot_chains_BD([loc_chain_0, loc_chain_1], label="Local Proposal", color="green")
plot_chains_BD([uniform_chain_0, uniform_chain_1], label="Uniform Proposal", color="orange")
plot_chains_BD([qe_chain_0, qe_chain_1], label="Qe Proposal", color="blue")
plot_chains_BD([cst_chain_link_0, cst_chain_link_1], label="CST Link Proposal", color="pink")
plot_chains_BD([cst_chain_relation_0, cst_chain_relation_1], label="CST Relation Proposal", color="purple")
plot_chains_BD([cst_chain_both_0, cst_chain_both_1], label="CST Both Proposal", color="red")
plt.ylabel("BD Action")
plt.xlabel("MCMC Step")
plt.legend()
plt.show()












"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)):
        calculated_BD = BD_action_model.get_energy(bitstring)
        expected_BD = calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False)
        print(f"Bitstring: {bitstring}, Calculated BD: {calculated_BD}, Expected BD: {expected_BD}")
        if not np.isclose(calculated_BD, expected_BD, atol=1e-10):
            print("Error: Calculated BD action does not match expected value!")
            print("as it is just first order approximation, this is fine...")
            # print(calculated_BD - expected_BD)
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False))
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=True, first_order_taylor=False))
            # print(calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=True))
"""



"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and TC_model.get_constraint_energy(bitstring) == 0:
        pass
    if not is_causal_matrix(bitstring_to_matrix(bitstring)) and TC_model.get_constraint_energy(bitstring) <0.99:
        print("Error: Invalid state has low constraint energy!")
        print(bitstring)
    print(f"Bitstring: {bitstring}, Constraint Energy: {TC_model.get_constraint_energy(bitstring)}, Is Causal: {is_causal_matrix(bitstring_to_matrix(bitstring))}")
"""

"""# generate all bitstrings of lengtt n
S = ["".join(i) for i in itertools.product("01", repeat=n)]
for bitstring in S:
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and BD_TC_model.get_constraint_energy(bitstring) == 0:
        pass
    if not is_causal_matrix(bitstring_to_matrix(bitstring)) and BD_TC_model.get_constraint_energy(bitstring) <0.99:
        print("Error: Invalid state has low constraint energy!")
        print(bitstring)
    
    expected_BD = calculate_action(bitstring_to_matrix(bitstring), epsilon=epsilon, stdim = 4, first_order_smearing=False, first_order_taylor=False)
    calculated_BD = BD_TC_model.get_energy(bitstring)
    if is_causal_matrix(bitstring_to_matrix(bitstring)) and not np.isclose(calculated_BD, expected_BD, atol=1e-10):
        print("Error: Calculated BD action does not match expected value for valid state!")
        print(f"Bitstring: {bitstring}, Calculated BD: {calculated_BD}, Expected BD: {expected_BD}")

"""