"""Analytical solver for the contamination process."""

import pathlib
from typing import NamedTuple
from .problem_setup import ContaminationProcess, PoissonProcess, SingletonPopulation


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
            self.jl_solver = jl.SingletonPopulation.Problem(ctmn_rate, event_rate, ctmn_period)
        else:
            from .analy.singleton_population import LoopSolver

            self.loop_solver = LoopSolver(self.ctnm_proc, self.event_proc)

    def __call__(self, n: int, l: int):  # noqa: E741
        case = Case(n, l)
        if not self.use_julia:
            return lambda T: self.loop_solver(case)(T)
        else:
            return lambda T: self.jl_solver(n, l)(T)
