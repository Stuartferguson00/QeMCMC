# Quick Start

A full run in three steps: define an energy model, build a quantum proposal, and sample a
Markov chain. See [notebooks/Basics/1_basic_QeMCMC.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/Basics/1_basic_QeMCMC.ipynb)
for a worked example.

## 1. Initialise an energy model

The energy model is the target distribution the sampler explores. Here we use `ModelMaker`
to build a random Ising instance; see [Energy Models](energy-model.md) to define your own.

```python
from qemcmc.model import ModelMaker

n = 6   # number of spins
model = ModelMaker(n, "Fully Connected Ising", name="Example Ising model").model
```

## 2. (Optional) Define coarse graining

Coarse graining proposes local multi-spin updates on predefined subgroups instead of
updating all spins at once. Each subgroup is a list of spin indices.

```python
from qemcmc.coarse_grain import CoarseGraining

cg = CoarseGraining(
    n=n,
    subgroups=[[0, 1, 2], [3, 4, 5]],
    subgroup_probs=[0.5, 0.5],
)
```

See [Coarse Graining](coarse-graining.md) for full details.

## 3. Create and run QeMCMC

QeMCMC provides two quantum proposal mechanisms:

- **`QuenchProposal`**: Evolves under a static transverse-field Hamiltonian (quench approach).
- **`AdiabaticProposal`**: Evolves under a time-dependent Hamiltonian with an adiabatic ramp and plateau. See [Adiabatic Approach](adiabatic.md).

`MCMCRunner` drives the chain using the Metropolis acceptance test at the chosen temperature.

```python
from qemcmc.sampler import QuenchProposal, AdiabaticProposal
from qemcmc.sampler.runners import MCMCRunner

steps = 300   # length of the Markov chain
temp = 0.1    # temperature of the system

runner = MCMCRunner(model, temp)

# Option A: Quench proposal (static Hamiltonian evolution)
quench_proposal = QuenchProposal(
    model,
    gamma=(0.3, 0.6),   # relative strength of the mixer Hamiltonian
    time=(1, 10),       # Hamiltonian simulation time
    # coarse_graining=cg,   # uncomment to use the subgroups from step 2
)

# Option B: Adiabatic proposal (time-dependent adiabatic dressing)
adiabatic_proposal = AdiabaticProposal(
    model,
    gamma=2.0,          # mixer weight
    alpha=2.0,          # adiabatic ramp duration
    kappa=3.0,          # plateau duration
    delta_t=0.3,        # Trotter step size
    # coarse_graining=cg,
)

chain = runner.run(quench_proposal, steps, name="QeMCMC", verbose=True)
```

`run` returns an `MCMCChain` holding the sampled states.
