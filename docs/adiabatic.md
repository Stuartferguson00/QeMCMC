# Adiabatic Approach

In addition to the standard quench-based approach, QeMCMC supports an **adiabatic dressing** approach for generating quantum proposals.

This approach is based on [Adiabatic dressing of quantum enhanced Markov chains](https://arxiv.org/abs/2603.28076) by _W. Hsieh, A. Orfi and D. Sels._


---

## Quench vs. Adiabatic Proposals

- **Quench approach (`QuenchProposal`)**: Evolves the initial computational basis state under a **time-independent** transverse-field Hamiltonian:
  $$H = \gamma H_{\text{mixer}} + (1 - \gamma) H_{\text{problem}}$$
  for an evolution time $t$. This follows the formulation by [Layden et al. (2023)](https://www.nature.com/articles/s41586-023-06095-4).

- **Adiabatic approach (`AdiabaticProposal`)**: Evolves the state under a **time-dependent** Hamiltonian:
  $$H(t) = h(t)\,\gamma\,H_{\text{mixer}} + H_{\text{problem}}$$
  where $h(t)$ is a dimensionless schedule function satisfying $h(0) = 0$ and $h(t_{\text{total}}) = 0$.

### Why Adiabatic Dressing?

By turning the transverse-field mixer on and off smoothly, the adiabatic theorem ensures that for very long evolution times, the quantum state returns close to the initial basis state. For intermediate evolution times, controlled **diabatic transitions** excite the system toward states with similar or lower energies in the target distribution. This produces candidate states that are non-local yet retain high Metropolis acceptance rates.

---

## The Schedule Function

The schedule $h(t)$ implemented in `AdiabaticCircuitMaker` consists of:

1. A **linear ramp-up** of duration $\alpha$ from $h=0$ to $h=1$.
2. A **plateau** of duration $\kappa$ held at $h=1$.
3. A **linear ramp-down** of duration $\alpha$ from $h=1$ to $h=0$.

The total evolution time is:
$$t_{\text{total}} = 2\alpha + \kappa$$

```
    h(t)
     1.0 |           +----------------+
         |          /                  \
         |         /                    \
         |        /                      \
     0.0 +-------+------------------------+-----> t
         0     alpha                2*alpha+kappa
```

The time-dependent evolution is approximated using second-order Trotterisation:
$$r = \max\left(1, \left\lfloor \frac{2\alpha + \kappa}{\Delta t} + 0.5 \right\rfloor\right)$$
where $\Delta t$ is the Trotter step size (`delta_t`).

---

## Using `AdiabaticProposal`

Create an `AdiabaticProposal` by passing the energy model along with the adiabatic hyperparameters:

```python
from qemcmc.model import ModelMaker
from qemcmc.sampler import AdiabaticProposal
from qemcmc.sampler.runners import MCMCRunner

# 1. Define model
n = 6
model = ModelMaker(n, "Fully Connected Ising", name="Example").model

# 2. Configure the adiabatic proposal
proposal = AdiabaticProposal(
    model,
    gamma=2.0,       # mixer strength factor
    alpha=2.0,       # adiabatic ramp duration
    kappa=3.0,       # plateau duration
    delta_t=0.3,     # Trotter step size
)

# 3. Run the Markov chain
runner = MCMCRunner(model, temp=0.1)
chain = runner.run(proposal, n_hops=300, name="Adiabatic QeMCMC", verbose=True)
```

### Hyperparameter Ranges

Like `QuenchProposal`, hyperparameters can be passed as `(min, max)` tuples to sample values uniformly at each MCMC step:

```python
proposal = AdiabaticProposal(
    model,
    gamma=(1.5, 3.0),
    alpha=(1.0, 3.0),
    kappa=(2.0, 4.0),
    delta_t=0.3,
)
```

---

## Combining with Coarse Graining

Adiabatic proposals work seamlessly with `CoarseGraining` to scale to larger spin systems:

```python
from qemcmc.coarse_grain import CoarseGraining
from qemcmc.sampler import AdiabaticProposal

# Partition a 12-spin system into 3 subgroups of 4 spins
cg = CoarseGraining(n=12, m=3)

proposal = AdiabaticProposal(
    model,
    gamma=2.0,
    alpha=2.0,
    kappa=3.0,
    delta_t=0.3,
    coarse_graining=cg,
)
```

---

## Summary of Proposal Types

| Feature | `QuenchProposal` | `AdiabaticProposal` |
|---|---|---|
| **Mechanism** | Static transverse-field quench | Time-dependent adiabatic dressing |
| **Circuit Maker** | `QuenchCircuitMaker` | `AdiabaticCircuitMaker` |
| **Hamiltonian** | $H = \gamma H_m + (1-\gamma) H_p$ | $H(t) = h(t) \gamma H_m + H_p$ |
| **Time Parameters** | `time` ($t$) | `alpha` ($\alpha$) and `kappa` ($\kappa$) |
| **Total Duration** | $t$ | $2\alpha + \kappa$ |
| **Trotter Steps** | Fixed `r` or $t / \Delta t$ | $\text{round}((2\alpha + \kappa) / \Delta t)$ |
| **Reference** | Layden et al. (2023) | Ferguson & Wallden / Hsieh et al. (2026) |

---

## Example Notebooks

- [notebooks/adiabatic/1_Adiabatic_QeMCMC.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/adiabatic/1_Adiabatic_QeMCMC.ipynb): Walkthrough comparing classical MCMC with adiabatic QeMCMC.
- [notebooks/adiabatic/2_Adiabatic_coarse_graining.ipynb](https://github.com/Stuartferguson00/QeMCMC/blob/main/notebooks/adiabatic/2_Adiabatic_coarse_graining.ipynb): Combining coarse-graining with adiabatic proposals.
