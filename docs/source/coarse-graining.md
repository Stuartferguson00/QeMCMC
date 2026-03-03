# Coarse Graining
To do coarse graining, a list of subgroups and it's corresponding probabilities along with the couplings list should be passed in as parameters when initializing an energy model. The subgroup list should contain lists of spin indices that belong to each subgroup. For example, for a system with 6 spins divided into 2 subgroups of 3 spins each, the subgroup list would be `[[0, 1, 2], [3, 4, 5]]`. 

NOTE: All spins must belong to at least one subgroup and subgroups may overlap (i.e., a spin may belong to multiple subgroups).

The `CircuitMaker` will then automatically build the evolution circuit to perform coarse graining based on these subgroups.


## Initializing an object
Coarse graining allows the sampler to propose local multi-spin updates on predefined subgroups, rather than updating all spins at once.
```python
from qemcmc.coarse_grain import CoarseGraining

cg = CoarseGraining(
    n=n,
    subgroups=[[0,1], [2,3,4], [1,3]],
    subgroup_probs=[0.3, 0.5, 0.2],
)
```
Each subgroup specifies a set of spin indices that may be updated together.
At each 