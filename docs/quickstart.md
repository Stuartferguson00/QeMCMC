# Quick Start

A full run in three steps: define an energy model, build a quantum proposal, and sample a
Markov chain. See [notebooks/Basics/basic_QeMCMC.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/Basics/basic_QeMCMC.ipynb)
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

See [Coarse Graining](coarse-graining.md) for the full details.

## 3. Create and run QeMCMC

`QeProposal` generates proposals via Trotterised quantum time evolution; `MCMCRunner` drives
the chain using the Metropolis acceptance test at the chosen temperature.

```python
from qemcmc.sampler import QeProposal
from qemcmc.sampler.runners import MCMCRunner

steps = 300   # length of the Markov chain
temp = 0.1    # temperature of the system

runner = MCMCRunner(model, temp)

quantum_proposal = QeProposal(
    model,
    gamma=(0.3, 0.6),   # relative strength of the mixer Hamiltonian
    time=(1, 10),       # Hamiltonian simulation time
    # coarse_graining=cg,   # uncomment to use the subgroups from step 2
)

chain = runner.run(quantum_proposal, steps, name="QeMCMC", verbose=True)
```

`run` returns an `MCMCChain` holding the sampled states. `gamma` and `time` may be single
floats or `(low, high)` ranges that are sampled per step.
