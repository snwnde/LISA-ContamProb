"""
This module provides a normal approximation approach to solving the contamination
probability problem.
"""

import abc
import pathlib
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Protocol,
    TypeVar,
    Literal,
    TypedDict,
    Unpack,
    Generic,
)
import logging
import numpy as np
import scipy.stats  # type: ignore[import]

if TYPE_CHECKING:
    from .problem_setup import (
        SCENARIO,
        PoissonProcess,
        ContaminationPeriodPopulation,
        ContaminationProcess,
        SingletonPopulation,
        UniformDistribution,
        ExponentialDistribution,
    )

_CTMN_POP = TypeVar("_CTMN_POP", bound="ContaminationPeriodPopulation", covariant=True)


class ApproxConfig(TypedDict):
    """The configuration for the approximation."""

    prob_method: Literal["by_hand", "recurrence"]
    """The probability calculation method."""

    max_k: int
    """The contamination number cut-off."""

    self_ctmn: bool
    """True if we are computing the self-contaminating case."""


class CtmnProcApprox(Protocol):
    def __init__(
        self,
        process: "PoissonProcess",
        contamination: _CTMN_POP,
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ): ...


class PDFResults(NamedTuple):
    """The results of the PDF calculation."""

    mean: float
    """The mean of the PDF."""

    variance: float
    """The variance of the PDF."""

    pdf: Callable
    """The PDF function."""


class SelfCtmnPDFResults(NamedTuple):
    """The results of the self-contaminating PDF calculation."""

    num_mean: float
    """The mean of the PDF of the numerator."""

    num_variance: float
    """The variance of the PDF of the numerator."""

    covariance: float
    """The covariance of the PDF of the numerator and denominator."""

    interval_mean: float
    """The mean of the PDF of the contamination interval length."""

    interval_variance: float
    """The variance of the PDF of the contamination interval length."""


log = logging.getLogger(__name__)


class SingletonPopulationApprox:
    def __init__(
        self,
        process: "PoissonProcess",
        contamination: "SingletonPopulation",
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ):
        self.process = process
        self.contamination = contamination
        self.scenario = scenario
        self.config = config

    def __call__(self, observation_time: float):
        self_ctmn = self.config.get("self_ctmn", False)
        if self_ctmn:
            raise NotImplementedError(
                "Self-contaminating case is not implemented for SingletonPopulationApprox"
            )
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


def _get_ctmn_frac(ctmn_rate: float, results: PDFResults, observation_time: float):
    log.info(f"results.mean: {results.mean}, results.variance: {results.variance}")
    gap_mean, gap_var = 1 / ctmn_rate, 1 / ctmn_rate**2
    n_estimate = observation_time / (results.mean + gap_mean)
    log.info(f"gap_mean: {gap_mean}, gap_var: {gap_var}, n_estimate: {n_estimate}")
    frac_mean = results.mean / (results.mean + gap_mean)
    frac_var = (1 / n_estimate) * (
        results.variance / (results.mean + gap_mean) ** 2
        + results.mean**2
        / (results.mean + gap_mean) ** 4
        * (results.variance + gap_var)
        - 2 * results.mean * results.variance / (results.mean + gap_mean) ** 3
    )
    log.info(f"frac_mean: {frac_mean}, frac_var: {frac_var}")
    mean = frac_mean * observation_time
    variance = frac_var * observation_time**2
    log.info(f"mean: {mean}, variance: {variance}")
    return mean, variance


def _get_self_ctmn_frac(
    ctmn_rate: float, results: SelfCtmnPDFResults, observation_time: float
):
    log.info(
        "results.interval_mean: %s, results.num_mean: %s, "
        "results.num_variance: %s, results.covariance: %s",
        results.interval_mean,
        results.num_mean,
        results.num_variance,
        results.covariance,
    )
    gap_mean, gap_var = 1 / ctmn_rate, 1 / ctmn_rate**2
    n_estimate = observation_time / (results.interval_mean + gap_mean)
    frac_mean = results.num_mean / (results.interval_mean + gap_mean)
    frac_var = (1 / n_estimate) * (
        results.num_variance / (results.interval_mean + gap_mean) ** 2
        + results.num_mean**2
        / (results.interval_mean + gap_mean) ** 4
        * (results.interval_variance + gap_var)
        - 2
        * results.num_mean
        * results.covariance
        / (results.interval_mean + gap_mean) ** 3
    )
    log.info(f"frac_mean: {frac_mean}, frac_var: {frac_var}")
    mean = frac_mean * observation_time
    variance = frac_var * observation_time**2
    log.info(f"mean: {mean}, variance: {variance}")
    return mean, variance


class _JuliaApprox(Generic[_CTMN_POP]):
    MODULE_NAME: Literal["Exponential", "Uniform"]

    def __init__(
        self,
        process: "PoissonProcess",
        contamination: _CTMN_POP,
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ):
        self.process = process
        self.contamination = contamination
        self.scenario = scenario
        self.config = config

        scenario_module = {
            "merged_interval": "MergedInterval",
            "reset_interval": "ResetInterval",
        }[scenario]

        import juliacall  # type: ignore[import]

        jl = juliacall.newmodule(__name__)
        approx_main = pathlib.Path(__file__).parent / "approx" / "ApproxMain.jl"
        jl.seval(f'include("{approx_main}")')
        self.jl = getattr(getattr(jl.ApproxMain, self.MODULE_NAME), scenario_module)

    @abc.abstractmethod
    def _get_pdf(self, observation_time: float): ...

    def _get_pdf_results(self, observation_time: float) -> PDFResults:
        prob = self._get_pdf(observation_time)
        # For numerical accuracy, we need to pass the integration upper bound
        # to the mean and variance functions if contamination periods are
        # drawn from a uniform distribution.
        args = (
            (self.config["max_k"] * self.contamination.upper,)
            if hasattr(self.contamination, "upper")
            else ()
        )
        mean = self.jl.mean(prob, *args)
        variance = self.jl.variance(prob, *args)
        return PDFResults(mean, variance, prob)

    def _get_self_ctmn_pdf_results(self, observation_time: float) -> SelfCtmnPDFResults:
        prob = self._get_pdf(observation_time)
        num_mean = self.jl.self_ctmn_num_mean(prob)
        num_variance = self.jl.self_ctmn_num_variance(prob)
        covariance = self.jl.self_ctmn_covariance(prob)
        interval_mean = self.jl.mean(prob)
        interval_variance = self.jl.variance(prob)
        return SelfCtmnPDFResults(
            num_mean, num_variance, covariance, interval_mean, interval_variance
        )

    def __call_ctmn__(self, observation_time: float):
        ctmn_rate = self.process.rate
        results = self._get_pdf_results(observation_time)
        log.info(f"results.mean: {results.mean}, results.variance: {results.variance}")
        return _get_ctmn_frac(ctmn_rate, results, observation_time)

    def __call_self_ctmn__(self, observation_time: float):
        ctmn_rate = self.process.rate
        results = self._get_self_ctmn_pdf_results(observation_time)
        log.info(
            "results.interval_mean: %s, results.num_mean: %s, "
            "results.num_variance: %s, results.covariance: %s",
            results.interval_mean,
            results.num_mean,
            results.num_variance,
            results.covariance,
        )
        return _get_self_ctmn_frac(ctmn_rate, results, observation_time)

    def __call__(self, observation_time: float):
        self_ctmn = self.config.get("self_ctmn", False)
        if self_ctmn:
            return self.__call_self_ctmn__(observation_time)
        else:
            return self.__call_ctmn__(observation_time)


class ExponentialDistributionApprox(_JuliaApprox):
    MODULE_NAME = "Exponential"

    def __init__(
        self,
        process: "PoissonProcess",
        contamination: "ExponentialDistribution",
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ):
        super().__init__(process, contamination, scenario, **config)
        self.contamination = contamination

    def _get_pdf(self, observation_time: float):
        ctmn_rate = float(self.process.rate)
        mean_ctmn = float(self.contamination.mean)
        obs_time = float(observation_time)
        del obs_time
        max_k = self.config.get("max_k", -1)
        if max_k < 0 and self.scenario != "reset_interval":
            raise ValueError(f"max_k must be set for {self.scenario} scenario")
        if self.config["prob_method"] == "by_hand":
            prob = self.jl.ProbByHand(ctmn_rate, mean_ctmn, max_k)
            return prob

        raise NotImplementedError


class UniformDistributionApprox(_JuliaApprox):
    MODULE_NAME = "Uniform"

    def __init__(
        self,
        process: "PoissonProcess",
        contamination: "UniformDistribution",
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ):
        super().__init__(process, contamination, scenario, **config)
        self.contamination = contamination

    def _get_pdf(self, observation_time: float):
        ctmn_rate = float(self.process.rate)
        max_ctmn = float(self.contamination.upper)
        obs_time = float(observation_time)
        del obs_time
        max_k = self.config.get("max_k", -1)
        if max_k < 0:
            raise ValueError(f"max_k must be set for {self.scenario} scenario")
        if self.config["prob_method"] == "by_hand":
            prob = self.jl.ProbByHand(ctmn_rate, max_ctmn, max_k)
            return prob
        else:
            raise NotImplementedError


class _DebugApprox:
    def __init__(
        self,
        mean: float,
        variance: float,
    ):
        self.mean = mean
        self.variance = variance

    def __call__(self, observation_time: float):
        del observation_time
        return self.mean, self.variance


class NormalApproximation:
    """A normal approximation to the contamination process."""

    def __init__(
        self,
        ctmn_proc: "ContaminationProcess",
        **config: Unpack[ApproxConfig],
    ):
        self.ctmn_proc = ctmn_proc
        self.ctmn_proc_approx = ctmn_proc.approx(**config)

    def __call__(self, observation_time: float):
        """Return the normal distribution approximation for the observation time."""
        mean, variance = self.ctmn_proc_approx(observation_time)
        return scipy.stats.norm(mean, np.sqrt(variance))

    def get_ctmn_interval_pdf(self, observation_time: float):
        """Return the contamination interval pdf for the observation time."""
        if isinstance(self.ctmn_proc_approx, _JuliaApprox):
            results = self.ctmn_proc_approx._get_pdf_results(observation_time)
            return results.pdf
        raise NotImplementedError("Only Julia approximations are supported.")


class DebugNormalApproximation:
    """A debug approximation to the contamination process.

    This is a simple approximation that takes the mean and variance
    of the distribution of the contamination intervals and computes
    the normal approximation from them.
    """

    def __init__(
        self,
        ctmn_proc: "ContaminationProcess",
        mean: float,
        variance: float,
    ):
        self.ctmn_proc = ctmn_proc
        self.mean = mean
        self.variance = variance

    def __call_ctmn__(self, observation_time: float):
        del observation_time
        return self.mean, self.variance
