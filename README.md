# Quantum-enhanced Markov Chain Monte Carlo Simulator

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Qiskit 2.2](https://img.shields.io/badge/Qiskit-2.2-cyan)](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.2)
[![PennyLane 0.44](https://img.shields.io/badge/PennyLane-0.44-pink)](https://docs.pennylane.ai/en/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


This is a lightweight research package for **Quantum-enhanced Markov Chain Monte Carlo** (QeMCMC) sampling over discrete spin/bitstring configurations.  


The implementation is inspired by the numerics in [Layden's work on QeMCMC](https://www.nature.com/articles/s41586-023-06095-4) and builds upon the foundations of the [pafloxy/quMCMC](https://github.com/pafloxy/quMCMC) repository.


## Features
- **Arbitrary Energy Models:** Define any classical Ising or QUBO-like model using a simple list of coupling tensors (example: 2D Ising `h`, `J` etc). A universal energy calculator handles arbitrary-order interactions
- **Automatic Hamiltonian Construction:** Build the corresponding **quantum Hamiltonian** based on the given couplings and run **Trotterised time evolution** with PennyLane's lightining qubit simulator
- **Coarse Graining**: Optionally use local updates on chosen subgroups of spins to scale proposals


## Installation

This project uses [`uv`](https://astral.sh/uv), an extremely fast Python package installer written in Rust, intended as a drop-in replacement for `pip` and `pip-tools`. Official installation instructions available at [astral.sh/uv](https://astral.sh/uv)

1.  **Install `uv`:**
    For macOS and Linux run:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment:**
    From the project's root directory, run:
    ```bash
    uv sync
    ```
    This will create a local `.venv` folder and install all required dependencies from `pyproject.toml` and `uv.lock`.

## Quick Start - Example 2D Ising Model


<!-- `model/energy_model.py`

```python
# Use our pre-defined base class to initialise an energy model
class EnergyModel():
    ...

# Or optionally define a problem specific energy model inheriting the base class EnergyModel
class your_energy_model(EnergyModel)
    ...
``` -->
### 1. Initialise an energy model
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

### 2. (Optional) Define coarse graining
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

### 3. Create and run QeMCMC
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

## Coarse Graining
To do coarse graining, a list of subgroups and it's corresponding probabilities along with the couplings list should be passed in as parameters when initializing an energy model. The subgroup list should contain lists of spin indices that belong to each subgroup. For example, for a system with 6 spins divided into 2 subgroups of 3 spins each, the subgroup list would be `[[0, 1, 2], [3, 4, 5]]`. 

NOTE: All spins must belong to at least one subgroup and subgroups may overlap (i.e., a spin may belong to multiple subgroups).

The `PennyLaneCircuitMaker` will then automatically build the evolution circuit to perform coarse graining based on these subgroups.


## Documentation

QeMCMC's documentation is available at [docs](#).

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors

This project was created by Stuart Ferguson and maintained by Feroz Hassan.

For questions, suggestions, or collaboration, please feel free to contact the authors:

-   S.A.Ferguson-3@sms.ed.ac.uk.
-   fhassan2@ed.ac.uk

## Acknowledgements

-   [pafloxy/quMCMC](https://github.com/pafloxy/quMCMC) for the foundational code.
-   [Quantum-enhanced Markov Chain Monte Carlo](https://www.nature.com/articles/s41586-023-06095-4) by David Layden et al.
-   [Quantum-enhanced MCMC for systems larger than your Quantum Computer](https://arxiv.org/abs/2405.04247) by S. Ferguson and P. Wallden.

---
![alt text](logo-qsl.jpeg)