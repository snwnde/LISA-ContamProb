"""Analytical solver for the contamination process."""

import pathlib
from typing import NamedTuple
import numpy as np
import scipy.special  # type: ignore[import]
from .problem_setup import ContaminationProcess, PoissonProcess, SingletonPopulation


def pow_step(base, exp):
    return np.heaviside(base, 1) * np.power(base, exp)


class Case(NamedTuple):
    """The case to consider.

    Attributes:
        n (int): The number of contamination arrivals.
        l (int): The number of events lost due to the contamination arrivals.
    """

    n: int
    l: int  # noqa: E741


class AnalyticalSolver:
    """An analytical solver for the contamination process."""

    def __init__(
        self,
        ctnm_proc: ContaminationProcess,
        event_proc: PoissonProcess,
        use_julia: bool = False,
    ) -> None:
        self.ctnm_proc = ctnm_proc
        self.event_proc = event_proc
        self.scenario = ctnm_proc.scenario
        self.use_julia = use_julia
        # Check if the contamination process is a singleton population
        if not isinstance(ctnm_proc.contamination, SingletonPopulation):
            raise NotImplementedError("Only singleton population is supported.")
        if use_julia:
            import juliacall  # type: ignore[import]

            jl = juliacall.newmodule(__name__)
            jl_file_path = (
                pathlib.Path(__file__).parent / "analy" / "SingletonPopulation.jl"
            )
            jl.seval(f'include("{jl_file_path}")')
            ctmn_rate, event_rate, ctmn_period = (
                ctnm_proc.process.rate,
                event_proc.rate,
                ctnm_proc.contamination.value,
            )
            self.jl_solver = jl.SingletonPopulation.Problem(
                ctmn_rate, event_rate, ctmn_period
            )
        else:
            from .analy.singleton_population import LoopSolver

            self.loop_solver = LoopSolver(self.ctnm_proc, self.event_proc)

    def __call__(self, n: int, l: int):  # noqa: E741
        case = Case(n, l)
        if not self.use_julia:
            return lambda T: self.loop_solver(case)(T)
        else:
            return lambda T: self.jl_solver(n, l)(T)


class SelfContamination:
    def __init__(
        self,
        event_proc: ContaminationProcess,
        use_julia: bool = False,
    ) -> None:
        self.event_proc = event_proc
        self.scenario = event_proc.scenario
        self.use_julia = use_julia
        if not isinstance(event_proc.contamination, SingletonPopulation):
            raise NotImplementedError("Only singleton population is supported.")
        self.tau = event_proc.contamination.value

    def __call_py__(self, n: int, k: int):
        def call(T):
            print([scipy.special.comb(k, j) for j in range(k + 1)])
            print([(1 - j * self.tau / T, k - 1) for j in range(k + 1)])
            print([pow_step(1 - j * self.tau / T, k - 1) for j in range(k + 1)])
            print(list(
                    (-1) ** j
                    * scipy.special.comb(k, j)
                    * pow_step(1 - j * self.tau / T, k - 1)
                    for j in range(k + 1)
                ))
            print(
                sum(
                    (-1) ** j
                    * scipy.special.comb(k, j)
                    * pow_step(1 - j * self.tau / T, k - 1)
                    for j in range(k + 1)
                )
            )
            print(pow_step(1 - (n + 1 - k) * self.tau / T, n + 1 - k))
            print(scipy.special.comb(n + 1, k))
            return (
                scipy.special.comb(n + 1, k)
                * pow_step(1 - (n + 1 - k) * self.tau / T, n + 1 - k)
                * sum(
                    (-1) ** j
                    * scipy.special.comb(k, j)
                    * pow_step(1 - j * self.tau / T, k - 1)
                    for j in range(k + 1)
                )
            )

        return call

    # def __call_py__(self, n: int, k: int):

    #     def pow_step_taylor(base, exp, order=2):
    #         """Approximate (1 - x)^n using a Taylor expansion for small x."""
    #         if base >= 0:
    #             x = 1 - base
    #             result = 1
    #             term = 1
    #             for i in range(1, order + 1):
    #                 term *= -exp * x / i
    #                 result += term
    #             return result
    #         else:
    #             return 0

    #     def call(T):
    #         if self.tau / T < 1e-3:  # Use Taylor expansion for small tau/T
    #             pow_func = pow_step_taylor
    #         else:  # Use standard pow_step otherwise
    #             pow_func = pow_step

    #         print(self.tau / T)
    #         print(list(
    #                 (-1) ** j
    #                 * scipy.special.comb(k, j)
    #                 * pow_func(1 - j * self.tau / T, k - 1)
    #                 for j in range(k + 1)
    #             ))

    #         return (
    #             scipy.special.comb(n + 1, k)
    #             * pow_func(1 - (n + 1 - k) * self.tau / T, n + 1 - k)
    #             * sum(
    #                 (-1) ** j
    #                 * scipy.special.comb(k, j)
    #                 * pow_func(1 - j * self.tau / T, k - 1)
    #                 for j in range(k + 1)
    #             )
    #         )

    #     return call

    #     # return (
    #     #     lambda T: scipy.special.comb(n + 1, k)
    #     #     * pow_step(1 - (n + 1 - k) * self.tau / T, n + 1 - k)
    #     #     * sum(
    #     #         (-1) ** j
    #     #         * scipy.special.comb(k, j)
    #     #         * pow_step(1 - j * self.tau / T, k - 1)
    #     #         for j in range(k + 1)
    #     #     )
    #     # )

    def __call__(self, n: int, k: int):
        if not self.use_julia:
            return self.__call_py__(n, k)
        else:
            raise NotImplementedError("Julia version is not implemented yet.")
