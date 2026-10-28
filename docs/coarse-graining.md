# Coarse Graining

Coarse graining lets the sampler propose local multi-spin updates on predefined subgroups,
rather than updating all spins at once. This keeps proposals tractable on larger systems.

## Defining subgroups

Create a `CoarseGraining` object with a list of subgroups and (optionally) their selection
probabilities. Each subgroup is a list of spin indices that are updated together.

- _Every spin must belong to at least one subgroup, and subgroups may overlap (a spin can
appear in multiple subgroups)._ 
- _`subgroup_probs` must sum to 1._
- _With the default `repeated=True`, all subgroups must be the same length. For
subgroups of differing sizes, pass `repeated=False`._

```python
from qemcmc.coarse_grain import CoarseGraining

n = 4
cg = CoarseGraining(
    n=n,
    subgroups=[[0, 1], [2, 3], [1, 3]],
    subgroup_probs=[0.3, 0.5, 0.2],
)
```

At each MCMC step a subgroup is sampled according to `subgroup_probs` (uniformly if omitted).


## Using it in a proposal

Pass the coarse-graining object to either `QuenchProposal` or `AdiabaticProposal` via `coarse_graining`; the circuit for each subgroup is built automatically.

```python
from qemcmc.sampler import QuenchProposal, AdiabaticProposal

# With QuenchProposal (quench approach)
proposal = QuenchProposal(model, gamma=(0.3, 0.6), time=(1, 10), coarse_graining=cg)

# Or with AdiabaticProposal (adiabatic dressing approach)
proposal = AdiabaticProposal(model, gamma=2.0, alpha=2.0, kappa=3.0, coarse_graining=cg)
```

See [notebooks/Basics/2_coarse_graining.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/Basics/2_coarse_graining.ipynb) for a quench coarse-graining tutorial and [notebooks/adiabatic/2_Adiabatic_coarse_graining.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/adiabatic/2_Adiabatic_coarse_graining.ipynb) for an adiabatic coarse-graining tutorial.
