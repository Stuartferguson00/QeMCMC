import numpy as np

from qemcmc.utils.helpers import validate_subgroups


class CoarseGraining:
    """
    CoarseGraining class to generate partitions of spins for quantum proposals.

    Parameters
    ----------
    n : int
        Number of spins in the system.
    subgroups : list[list[int]], optional
        A list of subgroups, where each subgroup is a list of spin indices. If None, the entire set of spins is treated as one subgroup.
    subgroup_probs : list[float], optional
        A list of probabilities corresponding to each subgroup, used for weighted random selection. Must sum to 1. If None, subgroups are selected uniformly at random.
    repeated : bool, optional
        If True, then multiple subgroups are run on the quantum computer in serial, if not then only one subgroup is selected at random and run on the quantum computer. Default is ``True``.
    """

    def __init__(self, n, subgroups=None, subgroup_probs=None, repeated=True):

        self._user_specified = subgroups is not None

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
        self.repeated = repeated
        
        if self.repeated and self._user_specified:
            if not np.all([len(s) == len(subgroups[0]) for s in subgroups]):
                raise NotImplementedError("If providing own subgroups, repeated must be set to False as it is not obvious how one performs repeated coarse-graining without knowledge of the subgroups. If length, l, of each subrgoup is equal, we continue assuming l = n//m.")
            else:
                print("Provided subgroups are all of same length, so repeated coarse-graining will be performed accross a random choice of subgroups. This means that the same variables may be chosen multiple times.")

    def sample(self) -> list[int]:
        """
        Randomly samples a subgroup according to the specified probability distribution.
        """
        idx = np.random.choice(len(self.subgroups), p=self.subgroup_probs)
        return self.subgroups[idx]

    def get_partitions(self, m: int) -> list[list[int]]:
        """
        Returns partitions of spins for sequential quantum updates.

        If the user provided explicit subgroups at initialisation, those are sampled accordingly.
        returned directly (all if ``repeated=True``, otherwise just the first).
        If no subgroups were specified, random disjoint partitions of
        approximate size n/m are generated.

        Parameters
        ----------
        m : int
            The number of partitions to generate. Ignored if user-specified subgroups are provided.
        """

        if self._user_specified:

            if self.repeated:
                # choose m subgroups from self.subgroups
                if len(self.subgroups) < m:
                    raise ValueError("Number of partitions, m, must be less than or equal to the number of provided subgroups.")
                choices = np.random.choice(np.arange(0,len(self.subgroups)), size = m, p = self.subgroup_probs, replace=False)
                
                chosen_subgroups = []
                for choice in choices:
                    chosen_subgroups.append(self.subgroups[choice])
                return chosen_subgroups
            else:
                chosen_subgroup = np.random.choice(self.subgroups, p=self.subgroup_probs)
                
                return [self.subgroups[chosen_subgroup]]

        spins = np.arange(self.n)
        np.random.shuffle(spins)
        # array_split handles uneven divisions gracefully
        chunks = [list(chunk) for chunk in np.array_split(spins, m)]
        if self.repeated:
            return chunks
        else:
            return [chunks[0]]
