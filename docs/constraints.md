# Constraints

Some problems only allow certain configurations. A `ConstraintModel` extends `EnergyModel`
with a constraint function: any state that fails the constraint is treated as having infinite
energy, so it is never accepted.

## Define a constraint model

Pass a `constraint_func` that takes a bitstring and returns `True` if the state is valid.
You also supply `constraint_couplings` (tensors describing the constraint, used to build the
quantum proposal) alongside the usual energy `couplings`.

```python
import numpy as np
from qemcmc.model import ModelMaker, ConstraintModel

n = 6
base_model = ModelMaker(n, "Fully Connected Ising", name="base").model

# Constraint: no two connected spins may differ
J_constraint = np.zeros((n, n))
J_constraint[0, 1] = J_constraint[1, 0] = 1

def constraint_func(bitstring: str) -> bool:
    total = 0
    for i in range(len(bitstring)):
        for j in range(i + 1, len(bitstring)):
            if bitstring[i] != bitstring[j]:
                total += J_constraint[i, j]
    return total == 0

model = ConstraintModel(
    n,
    couplings=base_model.couplings,
    constraint_couplings=[J_constraint],
    constraint_signs=[-1],
    constraint_func=constraint_func,
    name="constrained_ising",
)
```

## Run with the constrained runner

`ConstrainedMCMCRunner` rejects any proposed state that violates the constraint before the
Metropolis test. By default `run` returns the chain plus three rejection counts:
constraint, self (proposal unchanged), and Metropolis rejections. Pass
`return_rejections=False` to get just the chain.

```python
from qemcmc.sampler import QeProposal
from qemcmc.sampler.runners import ConstrainedMCMCRunner

runner = ConstrainedMCMCRunner(model, temp=0.1, reject_invalid=True)

proposal = QeProposal(model, gamma=(0.2, 0.3), time=(5, 10))

chain, n_constraint_rej, n_self_rej, n_mh_rej = runner.run(
    proposal, n_hops=200, name="QeMCMC", verbose=True
)
print("constraint rejection rate:", n_constraint_rej / 200)
```

See [notebooks/Constraints/constraints_tutorial.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/Constraints/constraints_tutorial.ipynb) for a worked example.
