# Example 2D Ising Model


##### 1. Initialise an energy model
In this step, we define a classical energy function over binary spin configurations.
This energy model is the target distribution that the QeMCMC sampler will explore.



```python
import numpy as np
from qemcmc.model import EnergyModel

n = 4
h = np.array([-1.0, 0.5, 0.0, 2.0])

# Symmetric J
J = np.array([
    [0.0, 1.2, 0.0, 0.0],
    [1.2, 0.0, -0.7, 0.0],
    [0.0, -0.7, 0.0, 0.4],
    [0.0, 0.0, 0.4, 0.0],
])

couplings = [h, J]

model = EnergyModel(n=n, couplings=couplings, name="my_ising")
```
Here, h represents local fields and J encodes pairwise interactions in a standard Ising formulation. Higher-order coupling tensors can also be supplied.

##### 2. (Optional) Define coarse graining
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
At each MCMC step, a subgroup is sampled according to subgroup_probs.

##### 3. Create and run QeMCMC
Finally, we initialise the quantum-enhanced Markov chain and generate a single proposal using simulated quantum time evolution.
```python
from qemcmc.qemcmc import QeMCMC

sampler = QeMCMC(
    model=model,
    gamma=(0.3, 0.6),     
    time=(1, 5),          
    temp=1.0,
    delta_time=0.8,
    coarse_graining=cg,
)

s = "01010"
s_prime = sampler.get_s_prime(s)
print("proposal:", s, "->", s_prime)
```
Here, each call to get_s_prime runs a quantum circuit to generate a proposal state conditioned on the current configuration.
