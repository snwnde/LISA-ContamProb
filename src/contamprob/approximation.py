"""
This module provides a normal approximation approach to solving the contamination
probability problem.
"""

from typing import TYPE_CHECKING, Protocol, TypeVar
import numpy as np
import scipy.stats  # type: ignore[import]

if TYPE_CHECKING:
    from .problem_setup import (
        SCENARIO,
        PoissonProcess,
        ContaminationPeriodPopulation,
        ContaminationProcess,
        SingletonPopulation,
        # UniformDistribution,
        # ExponentialDistribution,
    )

_CTMN_POP = TypeVar("_CTMN_POP", bound="ContaminationPeriodPopulation")


class CtmnProcApprox(Protocol):
    def __init__(
        self,
        process: "PoissonProcess",
        contamination: _CTMN_POP,
        scenario: "SCENARIO",
    ): ...


class SingletonPopulationApprox:
    def __init__(
        self,
        process: "PoissonProcess",
        contamination: "SingletonPopulation",
        scenario: "SCENARIO",
    ):
        self.process = process
        self.contamination = contamination
        self.scenario = scenario

    def __call__(self, observation_time: float):
        T = observation_time
        lam = self.process.rate
        tau = self.contamination.value
        mean = T * (1 - np.exp(-lam * tau))
        var = (
            2
            * T
            / lam
            * np.exp(-lam * tau)
            * (1 - (lam * tau + 1) * np.exp(-lam * tau))
        )
        return mean, var


class NormalApproximation:
    """A normal approximation to the contamination process."""

    def __init__(self, ctmn_proc: "ContaminationProcess"):
        self.ctmn_proc = ctmn_proc
        self.ctmn_proc_approx = ctmn_proc.approx

    def __call__(self, observation_time: float):
        """Return the normal distribution approximation for the observation time."""
        mean, variance = self.ctmn_proc_approx(observation_time)
        return scipy.stats.norm(mean, np.sqrt(variance))
