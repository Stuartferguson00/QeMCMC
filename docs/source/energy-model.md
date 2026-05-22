# Energy Models

An energy model defines a classical energy function over binary configurations which is the target distribution the QeMCMC sampler explores. The energy can be *any* function of the variables, but two common formulations are the **Ising model** and **QUBO** (Quadratic Unconstrained Binary Optimization). Both are *binary quadratic models*, and conversion between them is trivial (via $s = 2x - 1$).

In QeMCMC you describe a model by its coefficients, passed to `EnergyModel` as a list of
coupling tensors: a 1D array of linear terms and a 2D matrix of quadratic terms. Higher-order
tensors (3D, 4D, …) are also accepted for interactions beyond pairwise.

## Ising model

The Ising model is normally used in statistical mechanics and physics based problems. The $N$ variables
$s = [s_1, \dots, s_N]$ are spins taking values $s_i \in \{-1, +1\}$. The objective function is given by the energy:

$$E(s) = \sum_{i=1}^{N} h_i s_i + \sum_{i<j} J_{ij}\, s_i s_j, \qquad s_i \in \{-1, +1\}.$$

The linear coefficients $h_i$ are local fields and the quadratic coefficients
$J_{ij}$ are the coupling strengths between spins.

<!-- ```python
import numpy as np
from qemcmc.model import EnergyModel

n = 4
h = np.array([-1.0, 0.5, 0.0, 2.0])   # linear fields h_i
J = np.array([                         # quadratic couplings J_ij (symmetric, zero diagonal)
    [0.0,  1.2,  0.0, 0.0],
    [1.2,  0.0, -0.7, 0.0],
    [0.0, -0.7,  0.0, 0.4],
    [0.0,  0.0,  0.4, 0.0],
])

model = EnergyModel(n=n, couplings=[h, J], name="my_ising")   # model_type="ising" (default)
``` -->

## QUBO

QUBO  is generally used in computer science problems.
The variables $x = [x_1, \dots, x_N]$ are binary, taking values $x_i \in \{0, 1\}$. The objective function is given by the energy:

$$E(x) = \sum_{i} a_i x_i + \sum_{i<j} b_{ij}\, x_i x_j, \qquad x_i \in \{0, 1\},$$

where the $a_i$ are the linear coefficients and the $b_{ij}$ the quadratic coefficients. 

<!-- ```python
q = np.array([0.5, -1.0, 0.2, 0.0])   # linear coefficients a_i
Q = np.array([                         # quadratic coefficients b_ij (symmetric, zero diagonal)
    [ 0.0, -2.0,  0.0, 0.0],
    [-2.0,  0.0,  1.5, 0.0],
    [ 0.0,  1.5,  0.0, 0.8],
    [ 0.0,  0.0,  0.8, 0.0],
])

model = EnergyModel(n=n, couplings=[q, Q], name="my_qubo", model_type="binary")
``` -->

```{note}
Provide the quadratic term as a **full symmetric matrix with a zero diagonal** (not
upper-triangular): QeMCMC sums over all $i, j$ and halves the result, giving the $\sum_{i<j}$
above. Linear coefficients always go in the separate 1D array, never on the diagonal.
```

## Sign convention

By default QeMCMC negates each term (`cost_function_signs=[-1, -1]`), i.e. it evaluates

$$E(s) = -\sum_i h_i s_i - \sum_{i<j} J_{ij} s_i s_j$$ 

This is the convention for Boltzmann sampling $p(s) \propto e^{-E(s)/T}$.

## Quick models for testing

`ModelMaker` builds a small random Ising or QUBO instance so you don't have to write couplings
by hand while experimenting:

```python
from qemcmc.model import ModelMaker

model = ModelMaker(n, "Fully Connected Ising", name="test").model   # or "Fully Connected QUBO"
```

For constrained models, see [Constraints](constraints.md).
