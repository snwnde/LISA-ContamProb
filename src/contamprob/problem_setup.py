"""The signal contamination problem setting."""

from typing import NamedTuple, Protocol, Literal, Unpack, TypeVar, Generic
import numpy as np
import numpy.typing as npt

from .approximation import (
    ApproxConfig,
    CtmnProcApprox,
    SingletonPopulationApprox,
    ExponentialDistributionApprox,
    UniformDistributionApprox,
)

SCENARIO = Literal["constant_period", "merged_interval", "reset_interval"]
CPP = TypeVar("CPP", bound="ContaminationPeriodPopulation")


class PoissonProcess(NamedTuple):
    """A Poisson process."""

    rate: float
    """The rate of the process."""

    def __call__(self, duration: float, seed: None | int | np.random.Generator = None):
        rng1, rng2 = np.random.default_rng(seed).spawn(2)
        num = rng1.poisson(self.rate * duration)
        arrivals = rng2.uniform(0, duration, num)
        arrivals.sort()
        return arrivals


class ContaminationPeriodPopulation(Protocol):
    """A population of contamination periods."""

    def __call__(
        self, size: int, seed: None | int | np.random.Generator = None
    ) -> npt.NDArray:
        """Generate samples from the population."""

    def _get_approximator(
        self,
        process: PoissonProcess,
        scenario: SCENARIO,
        **config: Unpack[ApproxConfig],
    ) -> CtmnProcApprox:
        """Return the contamination process approximation."""


class UniformDistribution(NamedTuple):
    """A uniform distribution.

    The distribution is defined on the interval [0, upper].
    """

    upper: float
    """The upper bound of the distribution."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, self.upper, size)

    def _get_approximator(
        self,
        process: PoissonProcess,
        scenario: SCENARIO,
        **config: Unpack[ApproxConfig],
    ):
        """Return the contamination process approximation."""
        return UniformDistributionApprox(process, self, scenario, **config)


class ExponentialDistribution(NamedTuple):
    """An exponential distribution."""

    rate: float
    """The rate of the distribution."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        rng = np.random.default_rng(seed)
        return rng.exponential(1 / self.rate, size)

    @property
    def mean(self):
        """The mean of the distribution."""
        return 1 / self.rate

    @classmethod
    def from_scale(cls, scale: float):
        """Create an exponential distribution from the scale parameter."""
        return cls(rate=1 / scale)

    def _get_approximator(
        self,
        process: PoissonProcess,
        scenario: SCENARIO,
        **config: Unpack[ApproxConfig],
    ):
        """Return the contamination process approximation."""
        return ExponentialDistributionApprox(process, self, scenario, **config)


class SingletonPopulation(NamedTuple):
    """A population with a single value."""

    value: float
    """The value of the population."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        del seed
        return np.full(size, self.value)

    def _get_approximator(
        self,
        process: PoissonProcess,
        scenario: SCENARIO,
        **config: Unpack[ApproxConfig],
    ):
        """Return the contamination process approximation."""
        return SingletonPopulationApprox(process, self, scenario, **config)


class ContaminationProcess(NamedTuple, Generic[CPP]):
    """A contamination process."""

    process: PoissonProcess
    """The process generating the contamination periods."""

    contamination: CPP
    """The population of contamination periods."""

    scenario: SCENARIO
    """The contamination scenario."""

    def approx(self, **config: Unpack[ApproxConfig]):
        return self.contamination._get_approximator(
            self.process, self.scenario, **config
        )
