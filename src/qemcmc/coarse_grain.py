import numpy as np
from qemcmc.utils.helpers import validate_subgroups


class CoarseGraining:
    def __init__(self, n, subgroups=None, subgroup_probs=None):
        if subgroups is None:
            used = False
            subgroups = [list(range(n))]
            subgroup_probs = [1.0]
        else:
            used = True
            if subgroup_probs is None:
                subgroup_probs = [1.0 / len(subgroups)] * len(subgroups)

        validate_subgroups(subgroups=subgroups, subgroup_probs=subgroup_probs, n_spins=n)

        self.n = n
        self.subgroups = subgroups
        self.subgroup_probs = subgroup_probs
        self.used = used

    def sample(self, rng=np.random):
        idx = rng.choice(len(self.subgroups), p=self.subgroup_probs)
        return self.subgroups[idx]
