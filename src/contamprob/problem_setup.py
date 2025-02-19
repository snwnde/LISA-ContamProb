"""The signal contamination problem setting."""

from typing import NamedTuple, Protocol
import numpy as np
import numpy.typing as npt


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


class UniformDistribution(NamedTuple):
    """A uniform distribution."""

    lower: float
    """The lower bound of the distribution."""
    upper: float
    """The upper bound of the distribution."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper, size)


class ExponentialDistribution(NamedTuple):
    """An exponential distribution."""

    rate: float
    """The rate of the distribution."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        rng = np.random.default_rng(seed)
        return rng.exponential(1 / self.rate, size)

    @classmethod
    def from_scale(cls, scale: float):
        """Create an exponential distribution from the scale parameter."""
        return cls(rate=1 / scale)


class SingletonPopulation(NamedTuple):
    """A population with a single value."""

    value: float
    """The value of the population."""

    def __call__(self, size: int, seed: None | int | np.random.Generator = None):
        del seed
        return np.full(size, self.value)


class ContaminationProcess(NamedTuple):
    """A contamination process."""

    process: PoissonProcess
    """The process generating the contamination periods."""

    contamination: ContaminationPeriodPopulation
    """The population of contamination periods."""
