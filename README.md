# Quantum-enhanced Markov Chain Monte Carlo Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Qiskit 2.2](https://img.shields.io/badge/Qiskit-2.2-blueviolet)](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.2)
[![Qulacs 0.6.12](https://img.shields.io/badge/Qulacs-0.6.12-green)](https://docs.qulacs.org/en/v0.6.2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library provides a modular and extensible framework for implementing Quantum-enhanced Markov Chain Monte Carlo (QeMCMC) algorithms. It is designed to allow researchers and developers to easily define complex energy models and automatically generate piece-by-piece quantum circuits for time evolution.

The implementation is inspired by the numerics in [Layden's work on QeMCMC](https://www.nature.com/articles/s41586-023-06095-4) and builds upon the foundations of the [pafloxy/quMCMC](https://github.com/pafloxy/quMCMC) repository.

For a more advanced implementation that allows for simulations on systems larger than the available quantum computer, please see our other work: [CGQeMCMC](https://github.com/Stuartferguson00/CGQeMCMC).

## Key Features

-   **Flexible Energy Models:** Easily define any classical Ising or QUBO-like model using a simple list of coupling tensors (`h`, `J`, `L`, etc.). A universal energy calculator handles arbitrary-order interactions.
-   **Automatic Hamiltonian Construction:** The framework automatically converts your classical energy model into a quantum Hamiltonian operator, with support for **Qiskit** (`SparsePauliOp`) and ~~**Qulacs** (`Observable`)~~ (Qulacs support is discontinued and is deprecated for use with the latest versions of Qiskit).
-   **Optimized Circuit Building:** Instead of relying on slow, high-level gates, the `CircuitMaker` automatically constructs a high-performance, piece-by-piece time-evolution circuit using Trotterization.
-   **Backend Agnostic:** Core logic is designed to be compatible with multiple quantum computing frameworks.

NOTE: This library is built with flexibility and convenience in mind rather than performance or efficiency.

## Installation

This project uses [`uv`](https://astral.sh/uv), an extremely fast Python package installer written in Rust, intended as a drop-in replacement for `pip` and `pip-tools`.

1.  **Install `uv`:**
    Follow official installation instructions at [astral.sh/uv](https://astral.sh/uv). For macOS and Linux run:

    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

2.  **Create the Virtual Environment:**
    From the project's root directory, run the following command:
    ```bash
    uv sync
    ```
    This will create a local `.venv` folder and install all required dependencies from `pyproject.toml` and `uv.lock`.

## Quick Start

The library makes it simple to define a problem and generate the corresponding evolution circuit. Here is a quick example of a workflow:

`energy_models.py`

```python
# Pre-defined abstract base class for energy models. Initializes with a couplings list.
class EnergyModel(abc.ABC):
    ...

# Define your energy model here inheriting base class EnergyModel
class your_energy_model(EnergyModel)
    ...
```

`main.py`

```python
import numpy as np
from qemcmc.energy_models import your_energy_model # your defined energy model
from qemcmc.circuit_maker import CircuitMaker

# 1. Define the physics of your problem with a couplings list
h = np.array([-1.0, -2.0, -3.0])  # Linear terms
J = np.array([  # Quadratic terms
    [0.0, 0.5, 0.0],
    [0.5, 0.0, -1.5],
    [0.0, -1.5, 0.0]
])
my_couplings = [h, 0.5 * J] # Use 0.5*J to match standard Ising convention

# 2. Create an instance of your energy model
my_model = your_energy_model(couplings=my_couplings)

# 3. Instantiate the CircuitMaker with your model
# The evolution circuit is built automatically
circuit_maker = CircuitMaker(model=my_model, gamma=0.5, time=4.0)

# 4. Generate a new state proposal from an initial state
initial_state = "101"
proposed_state = circuit_maker.get_state_obtained_binary(initial_state)

print(f"Evolved from '{initial_state}' to '{proposed_state}'.")
print("\n--- Circuit Info ---")
print(circuit_maker.evolution_circuit)
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors and Contact

This project was created by Stuart Ferguson with contributions from Feroz Hassan.

For questions, suggestions, or collaboration, please feel free to contact the authors:

-   S.A.Ferguson-3@sms.ed.ac.uk.
-   F.M.Hassan@sms.ed.ac.uk

## Acknowledgements

-   [pafloxy/quMCMC](https://github.com/pafloxy/quMCMC) for the foundational code.
-   [Quantum-enhanced Markov Chain Monte Carlo](https://www.nature.com/articles/s41586-023-06095-4) by David Layden et al.
-   [Quantum-enhanced MCMC for systems larger than your Quantum Computer](https://arxiv.org/abs/2405.04247) by S. Ferguson and P. Wallden.
-   The [Qulacs](https://quantum-journal.org/papers/q-2021-10-06-559/) high-performance quantum circuit simulator.
