# code that, given a subgroup of relations, enumerates all possible proposals that could be made by the QeMCMC algorithm, and then checks which of these are valid (i.e. satisfy the contrsinats) and which are not)


import argparse
import sys
import json
import os
import pickle
import datetime
from matplotlib_inline import config
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
import time

def main():
    
    Cs = np.arange(4,9, 1)
    #relation_move_valid = []
    for m_ in ["nr","sqrt",3]:
        invalid_proposals_C = []
        total_proposals_C = []
        for C in Cs:
            print(f"========== Starting analysis for m={m_} ==========")
        
            n = C * (C - 1) // 2 # Number of (Qu)bits needed to represent the spacetime
            print("cardinality of CS:", C)
            print("number of bits needed to represent the spacetime:", n)
            if m_ is "nr":
                m = n
            elif m_ is "sqrt":
                m = int(np.sqrt(n))
            else:
                m = m_
            print("m_:", m_)
            print("m:", m)


            def constraint_checker_func(bitstring: str) -> bool:
                return is_causal_matrix(bitstring_to_matrix(bitstring))

        
            initial_states = ["0"*n, "1"*n]#, get_random_state(n)]

            cg = CoarseGraining(n, repeated=True)
            subgroups = cg.get_partitions(m = m)
            print(f"Number of possible subgroups: {len(subgroups)}")
            print(f"Subgroup[0]: {subgroups[0]}")

            num_subgroups = min(50, len(subgroups)) 
            test_subgroups = [subgroups[i] for i in range(num_subgroups)]
            # get all bitstrings of length test_subgroup
            
            
            is_len_invalid = []
            for initial_state in initial_states:
                valid_proposals = []
                invalid_proposals = []
                total_proposals = []
                for test_subgroup in test_subgroups:
                    bitstrings = [''.join(seq) for seq in itertools.product('01', repeat=len(test_subgroup))]
                    for bitstring in bitstrings:
                        full_bitstring = list(initial_state)
                        for i, ind in enumerate(test_subgroup):
                            full_bitstring[ind] = bitstring[i]
                        full_bitstring = ''.join(full_bitstring)
                        if constraint_checker_func(full_bitstring):
                            valid_proposals.append(full_bitstring)
                        else:
                            invalid_proposals.append(full_bitstring)
                    
                    total_proposals.append(2**len(test_subgroup))

                is_len_invalid.append(len(invalid_proposals))
            total_proposals_C.append(np.sum(total_proposals))
            invalid_proposals_C.append(np.array(is_len_invalid))
            
            # is_relation_move_valid = []
            # for initial_state in initial_states:
            #     relation_move_valid_ = []
            #     for test_subgroup in relation_subgroups:
            #         bitstrings = [''.join(seq) for seq in itertools.product('01', repeat=len(test_subgroup))]
            #         for bitstring in bitstrings:
            #             full_bitstring = list(initial_state)
            #             for i, ind in enumerate(test_subgroup):
            #                 full_bitstring[ind] = bitstring[i]
            #             full_bitstring = ''.join(full_bitstring)
            #             if constraint_checker_func(full_bitstring):
            #                 relation_move_valid_.append(full_bitstring)
            #     is_relation_move_valid.append(len(relation_move_valid_))
            # relation_move_valid.append(is_relation_move_valid)




        invalid_proposals_C = np.array(invalid_proposals_C)
        total_proposals_C = np.array(total_proposals_C)
        print("Cs:", Cs)
        print("Invalid proposals for each C:", invalid_proposals_C)
        print("invalid_proposals_C[:,0]:", invalid_proposals_C[:,0])
        print("Total proposals for each C:", total_proposals_C)
        print(Cs)
        if m_ is "nr":
            plt.plot(Cs, total_proposals_C - np.mean(invalid_proposals_C[:,:], axis = 1), label=f'valid Proposals m = num relations')
        elif m_ is "sqrt":
            plt.plot(Cs, total_proposals_C - np.mean(invalid_proposals_C[:,:], axis = 1), label=f'valid Proposals m = sqrt(num relations)')
        else:
            plt.plot(Cs, total_proposals_C - np.mean(invalid_proposals_C[:,:], axis = 1), label=f'valid Proposals m {m}')
        #plt.plot(Cs, total_proposals_C - invalid_proposals_C[:,1], label='valid Proposals (initial state 1)')
        #plt.plot(Cs, invalid_proposals_C[:,2], label='valid Proposals (random initial state)')
        #plt.plot(Cs, total_proposals_C, label='Total Proposals')

        #print("relation_move_valid:", relation_move_valid)
        #plt.plot(Cs, np.sum(relation_move_valid, axis = 1), label='valid Proposals (relation moves)')

        










    plt.xlabel('C')
    plt.ylabel('Number of Proposals')
    plt.title('Number of Invalid Proposals vs C')
    plt.legend()
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()
