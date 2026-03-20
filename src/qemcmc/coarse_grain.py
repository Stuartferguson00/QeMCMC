# Internal
from qemcmc.utils.helpers import validate_subgroups

# External
import numpy as np


class CoarseGraining:
    def __init__(self, n, subgroups=None, subgroup_probs=None):
        if subgroups is None:
            subgroups = [list(range(n))]
            subgroup_probs = [1.0]
        else:
            if subgroup_probs is None:
                subgroup_probs = [1.0 / len(subgroups)] * len(subgroups)

        validate_subgroups(subgroups=subgroups, subgroup_probs=subgroup_probs, n_spins=n)

        self.n = n
        self.subgroups = subgroups
        self.subgroup_probs = subgroup_probs

    def sample(self, rng=np.random):
        idx = rng.choice(len(self.subgroups), p=self.subgroup_probs)
        return self.subgroups[idx]

    def get_partitions(self, m: int, rng=np.random):
        """
        Randomly partitions the n spins into m disjoint subgroups.
        Sizes will be approximately n/m.
        """
        spins = np.arange(self.n)
        rng.shuffle(spins)
        # array_split handles uneven divisions gracefully
        return [list(chunk) for chunk in np.array_split(spins, m)]

