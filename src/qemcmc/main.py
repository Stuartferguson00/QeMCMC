import itertools
import os
from tqdm import tqdm
import pickle
import qemcmc
from qemcmc.utils.helpers import *
from qemcmc.model.model_maker import ModelMaker
from qemcmc.model.energy_model import EnergyModel
from typing import List
import numpy as np
import dimod
from qemcmc.sampler.qe_proposal import QeProposal
from qemcmc.sampler.runners import MCMCRunner
import time
# Basic helper code to initialise a list Ising models of type required by cgqemcmc
# Once created, Models are pickled so they can be easily accessed later.
for n_spins in [10,]:#np.arange(4,20):




    
    str_nspins = str(n_spins).zfill(3)


    
    model = ModelMaker(n_spins, "Fully Connected Ising", f"{str_nspins}").model
    
    model.lowest_energy = model.get_lowest_energy()

    proposer = QeProposal(
        model=model,
        gamma=(0.3, 0.6),
        time=(1, 20),
    )

    start = time.time()
    runner = MCMCRunner(model=model, temp=0.1)
    runner.run(proposer, 100)
    end = time.time()

    print("took:", end - start, "seconds")

    

