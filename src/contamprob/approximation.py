"""
This module provides a normal approximation approach to solving the contamination
probability problem.
"""

import abc
import pathlib
from typing import (
    TYPE_CHECKING,
    Callable,
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


class CtmnProcApprox(Protocol):
    def __init__(
        self,
        process: "PoissonProcess",
        contamination: _CTMN_POP,
        scenario: "SCENARIO",
        **config: Unpack[ApproxConfig],
    ): ...


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
    def _get_pdf_results(
        self, observation_time: float
    ) -> tuple[float, float, float, Callable]: ...

    def __call__(self, observation_time: float):
        ctmn_rate = self.process.rate
        pdf_mean, pdf_var, pdf_avg_k, _ = self._get_pdf_results(observation_time)
        log.info(f"pdf_mean: {pdf_mean}, pdf_var: {pdf_var}, pdf_avg_k: {pdf_avg_k}")
        gap_mean, gap_var = 1 / ctmn_rate, 1 / ctmn_rate**2
        n_estimate = observation_time / (pdf_mean + gap_mean)
        log.info(f"gap_mean: {gap_mean}, gap_var: {gap_var}, n_estimate: {n_estimate}")
        frac_mean = pdf_mean / (pdf_mean + gap_mean)
        frac_var = (1 / n_estimate) * (
            pdf_var / (pdf_mean + gap_mean) ** 2
            + pdf_mean**2 / (pdf_mean + gap_mean) ** 4 * (pdf_var + gap_var)
            - 2 * pdf_mean**2 * pdf_var / (pdf_mean + gap_mean) ** 3
        )
        log.info(f"frac_mean: {frac_mean}, frac_var: {frac_var}")
        mean = frac_mean * observation_time
        variance = frac_var * observation_time**2
        log.info(f"mean: {mean}, variance: {variance}")
        return mean, variance


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

    def _get_pdf_results(
        self, observation_time: float
    ) -> tuple[float, float, float, Callable]:
        ctmn_rate = float(self.process.rate)
        mean_ctmn = float(self.contamination.mean)
        obs_time = float(observation_time)
        del obs_time
        max_k = max_k = self.config.get("max_k", -1)
        if max_k < 0 and self.scenario != "reset_interval":
            raise ValueError(f"max_k must be set for {self.scenario} scenario")
        if self.config["prob_method"] == "by_hand":
            prob = self.jl.ProbByHand(ctmn_rate, mean_ctmn, max_k)
            mean = self.jl.mean(prob)
            variance = self.jl.variance(prob)
            avg_k = self.jl.avg_k(prob)
        else:
            raise NotImplementedError
        return mean, variance, avg_k, prob


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

    def _get_pdf_results(
        self, observation_time: float
    ) -> tuple[float, float, float, Callable]:
        ctmn_rate = float(self.process.rate)
        max_ctmn = float(self.contamination.upper)
        obs_time = float(observation_time)
        del obs_time
        max_k = max_k = self.config.get("max_k", -1)
        if max_k < 0:
            raise ValueError(f"max_k must be set for {self.scenario} scenario")
        if self.config["prob_method"] == "by_hand":
            prob = self.jl.ProbByHand(ctmn_rate, max_ctmn, max_k)
            # Either we provide max_k * max_ctmn here either we leave it to the julia code,
            # which will use the default value of Inf. The integration works fine
            # with Inf, but badly with obs_time.
            mean = self.jl.mean(prob, max_k * max_ctmn)
            variance = self.jl.variance(prob, max_k * max_ctmn)
            avg_k = self.jl.avg_k(prob)
        else:
            raise NotImplementedError
        return mean, variance, avg_k, prob


class NormalApproximation:
    """A normal approximation to the contamination process."""

    def __init__(
        self, ctmn_proc: "ContaminationProcess", **config: Unpack[ApproxConfig]
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
            _, _, _, pdf = self.ctmn_proc_approx._get_pdf_results(observation_time)
            return pdf
        raise NotImplementedError("Only Julia approximations are supported.")
