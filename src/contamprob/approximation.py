"""
This module provides a normal approximation approach to solving the contamination
probability problem.
"""

import numpy as np
import scipy.stats  # type: ignore[import]
from .problem_setup import (
    ContaminationProcess,
    SingletonPopulation,
    # UniformDistribution,
    # ExponentialDistribution,
)


class NormalApproximation:
    """A normal approximation to the contamination process."""

    def __init__(self, ctmn_proc: ContaminationProcess):
        self.ctmn_proc = ctmn_proc

    def _get_mean_var(self, observation_time: float):
        T = observation_time
        proc = self.ctmn_proc.process
        ctmn = self.ctmn_proc.contamination
        lam = proc.rate
        if isinstance(ctmn, SingletonPopulation):
            tau = ctmn.value  # type: ignore[attr-defined]
            # Weirdly mypy thinks ctmn is Never
            mean = T * (1 - np.exp(-lam * tau))
            var = (
                2
                * T
                / lam
                * np.exp(-lam * tau)
                * (1 - (lam * tau + 1) * np.exp(-lam * tau))
            )
        else:
            raise NotImplementedError(
                "The contamination period population is not supported."
            )
        return mean, var

    def __call__(self, observation_time: float):
        """Return the normal distribution approximation for the observation time."""
        mean, variance = self._get_mean_var(observation_time)
        return scipy.stats.norm(mean, np.sqrt(variance))
