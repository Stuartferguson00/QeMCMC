# Energy Models

To use the QeMCMC sampler, we first need to define a classical energy function over binary spin configurations. This energy model is the target distribution that the sampler will explore.

`model/energy_model.py`
```python
# Use our pre-defined base class to initialise an energy model
class EnergyModel():
    ...

# Or optionally define a problem specific energy model inheriting the base class EnergyModel
class your_energy_model(EnergyModel)
    ...
```