"""Simulation of contamination probabilities."""

import bisect
import pathlib
from collections.abc import Sequence, Mapping
from typing import Literal, NamedTuple, Self, Protocol, cast
import numpy as np
import numpy.typing as npt
from .problem_setup import ContaminationProcess, PoissonProcess


# Protocols
class _Interval(Protocol):
    start: float
    stop: float


class _DisjointUnion(Protocol):
    intervals: Sequence[_Interval]


# Julia Intervals
class JuliaInterval(_Interval): ...


class JuliaDisjointUnion(_DisjointUnion):
    intervals: Sequence[JuliaInterval]


# Interval Class
class Interval(NamedTuple):
    """An interval."""

    start: float
    """The start of the interval."""

    stop: float
    """The stop of the interval."""

    @classmethod
    def build(cls, start: float, stop: float) -> Self:
        """Create an interval from a start and a stop."""
        assert start <= stop, (
            "The start of the interval must be less than or equal to the stop."
        )
        return cls(start, stop)

    @classmethod
    def from_start(cls, start: float, duration: float) -> Self:
        """Create an interval from a start and a duration."""
        return cls.build(start, start + duration)

    @classmethod
    def from_end(cls, stop: float, duration: float) -> Self:
        """Create an interval from a stop and a duration."""
        return cls.build(stop - duration, stop)

    @classmethod
    def from_julia(cls, interval: JuliaInterval) -> Self:
        """Create an interval from a Julia object."""
        return cls(interval.start, interval.stop)

    @property
    def length(self) -> float:
        """Return the length of the interval."""
        return self.stop - self.start

    def capped(self, cap: float) -> Self:
        """Cap the interval."""
        return type(self)(self.start, min(self.stop, cap))

    def floored(self, floor: float) -> Self:
        """Floor the interval."""
        return type(self)(max(self.start, floor), self.stop)

    def contains(self, element: float) -> bool:
        """Check if an element is contained in the interval."""
        return self.start <= element <= self.stop

    def is_disjoint_with(self, other: Self) -> bool:
        """Check if two intervals are disjoint."""
        return self.stop < other.start or other.stop < self.start

    def merge_with(self, other: Self, reset_mode: bool = False) -> Self:
        """Compute the union of two non-disjoint intervals."""
        assert not self.is_disjoint_with(other), "The intervals must not be disjoint."
        if not reset_mode:
            return type(self)(min(self.start, other.start), max(self.stop, other.stop))
        if self.start <= other.start:
            return type(self)(self.start, other.stop)
        return type(self)(other.start, self.stop)


# Union Classes
class Union:
    """A union of intervals."""

    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = intervals

    def add_interval(self, other: Interval):
        """Add an interval to the union."""
        self.intervals.append(other)

    def contains(self, element: float) -> bool:
        """Check if an element is contained in the union."""
        return any(interval.contains(element) for interval in self.intervals)

    def sort(self):
        """Sort the intervals according to the start time."""
        self.intervals = sorted(self.intervals, key=lambda interval: interval.start)


class DisjointUnion(Union):
    """A union of disjoint intervals."""

    def add_interval(self, other: Interval, reset_mode: bool = False):
        """Add an interval to the union. If it overlaps with existing intervals, merge them.

        Caution: This method assumes that the intervals are sorted by start time.
        """
        intervals = self.intervals.copy()
        idx = bisect.bisect_left(intervals, other)
        if idx > 0 and not intervals[idx - 1].is_disjoint_with(other):
            idx -= 1

        while idx < len(intervals) and not intervals[idx].is_disjoint_with(other):
            other = other.merge_with(intervals.pop(idx), reset_mode=reset_mode)
        intervals.insert(idx, other)
        self.intervals = intervals

    @property
    def length(self) -> float:
        """Return the length of the union."""
        return sum(interval.length for interval in self.intervals)

    @property
    def is_disjoint_with(self) -> bool:
        """Check if the intervals are disjoint."""
        return self.is_sorted and all(
            self.intervals[i].is_disjoint_with(self.intervals[i + 1])
            for i in range(len(self.intervals) - 1)
        )

    @property
    def is_sorted(self) -> bool:
        """Check if the intervals are sorted."""
        return all(
            self.intervals[i].start <= self.intervals[i + 1].start
            for i in range(len(self.intervals) - 1)
        )

    @classmethod
    def from_julia(cls, disj_union: JuliaDisjointUnion) -> Self:
        """Create a disjoint union from a Julia object."""
        return cls([Interval.from_julia(interval) for interval in disj_union.intervals])


# class _SelfCtmnSimulationResult(Protocol):
#     ctmn_hierarchy: Mapping[
#         int, Mapping[Literal["culprits", "victims"], npt.NDArray[np.int_]]
#     ]
#     culprits: npt.NDArray[np.float64]
#     victims: npt.NDArray[np.float64]


class SelfCtmnSimulationResult(NamedTuple):
    """The result of a simulation."""

    ctmn_hierarchy: Mapping[
        int, Mapping[Literal["culprits", "victims"], npt.NDArray[np.int_]]
    ]
    """The hierarchy of contamination intervals."""
    culprits: npt.NDArray[np.float64]
    """The event arrivals that are culprits."""
    victims: npt.NDArray[np.float64]
    """The event arrivals that are contaminated."""


# Simulation Result Classes
class _SimulationResult(Protocol):
    event_arrivals: None | npt.NDArray[np.float64]
    ctmn_arrivals: npt.NDArray[np.float64]
    ctmn_periods: npt.NDArray[np.float64]
    ctmn_intervals: None | _DisjointUnion
    ctmn_length: float
    contaminated_events: None | npt.NDArray[np.float64]
    ctmn_int_categories: Mapping[int, int]
    self_ctmn_results: None | SelfCtmnSimulationResult


class JuliaSimulationResult(_SimulationResult):
    event_arrivals: None | npt.NDArray[np.float64]
    ctmn_arrivals: npt.NDArray[np.float64]
    ctmn_periods: npt.NDArray[np.float64]
    ctmn_intervals: None | JuliaDisjointUnion
    ctmn_length: float
    contaminated_events: None | npt.NDArray[np.float64]
    ctmn_int_categories: dict[int, int]
    self_ctmn_results: None | SelfCtmnSimulationResult


class SimulationResult(NamedTuple):
    """The result of a simulation."""

    event_arrivals: None | npt.NDArray[np.float64]
    """The arrival times of the events."""
    ctmn_arrivals: npt.NDArray[np.float64]
    """The arrival times of the contamination periods."""
    ctmn_periods: npt.NDArray[np.float64]
    """The durations of the contamination periods."""
    ctmn_intervals: DisjointUnion | None
    """The contamination intervals."""
    ctmn_length: float
    """The total length of the contamination intervals."""
    contaminated_events: None | npt.NDArray[np.float64]
    """The events that are contaminated."""
    ctmn_int_categories: dict[int, int]
    """The number of contamination intervals per number of contamination arrivals."""
    self_ctmn_results: None | SelfCtmnSimulationResult
    """The results of the self-contamination simulation."""

    @classmethod
    def from_julia(cls, result: JuliaSimulationResult) -> Self:
        """Create a simulation result from a Julia object."""
        return cls(
            result.event_arrivals,
            result.ctmn_arrivals,
            result.ctmn_periods,
            (
                None
                if result.ctmn_intervals is None
                else DisjointUnion.from_julia(result.ctmn_intervals)
            ),
            result.ctmn_length,
            result.contaminated_events,
            result.ctmn_int_categories,
            result.self_ctmn_results,
        )


# Simulator Classes
class _Simulator:
    def __init__(
        self,
        ctmn_proc: ContaminationProcess,
        event_proc: PoissonProcess | None = None,
        collect_events: bool = False,
        collect_stats: bool = False,
        k_cutoff: int | None = None,
        use_julia: bool = True,
        with_self_ctmn: bool = False,
    ) -> None:
        self.ctmn_proc = ctmn_proc
        self.event_proc = event_proc
        self.scenario = ctmn_proc.scenario
        self.collect_events = collect_events
        if event_proc is None and collect_events:
            raise ValueError(
                "Collecting events is not supported without an event process."
            )
        self.collect_stats = collect_stats
        self.k_cutoff = k_cutoff
        self.use_julia = use_julia
        self.with_self_ctmn = with_self_ctmn

        if use_julia:
            import juliacall  # type: ignore[import]

            jl = juliacall.newmodule(__name__)
            simu_core_jl = pathlib.Path(__file__).with_name("SimuCore.jl")
            jl.seval(f'include("{simu_core_jl}")')
            self.jl = jl

    def _generate_data(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        rngs = np.random.default_rng(seed).spawn(3)
        event_arrivals = (
            self.event_proc(observation_time, rngs[0])
            if self.event_proc and self.collect_events
            else None
        )
        ctmn_arrivals = self.ctmn_proc.process(observation_time, rngs[1])
        ctmn_periods = self.ctmn_proc.contamination(len(ctmn_arrivals), rngs[2])
        return event_arrivals, ctmn_arrivals, ctmn_periods


class Simulator(_Simulator):
    """Simulator for contamination probabilities for the merged interval scenario."""

    def __call_cst_period__(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        T = observation_time
        event_arrivals, ctmn_arrivals, ctmn_periods = self._generate_data(
            observation_time, seed
        )
        ctmn_period = ctmn_periods[0]
        if not all(ctmn_periods == ctmn_period):
            raise ValueError("The contamination periods must be constant.")
        if self.collect_events:
            raise NotImplementedError(
                "Collecting contaminated events is not supported."
            )

        if self.with_self_ctmn:
            raise NotImplementedError(
                "Self-contamination is not implemented for the constant period scenario yet."
            )

        separation = np.diff(ctmn_arrivals, append=T)
        separation[separation > ctmn_period] = ctmn_period
        ctmn_length = np.sum(separation)
        result = SimulationResult(
            event_arrivals,
            ctmn_arrivals,
            ctmn_periods,
            None,
            ctmn_length,
            None,
            {},
            None,
        )
        return result

    def __call_julia__(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        event_arrivals, ctmn_arrivals, ctmn_periods = self._generate_data(
            observation_time, seed
        )

        result = self.jl.SimuCore.ctmn_simulate(
            event_arrivals,
            ctmn_arrivals,
            ctmn_periods,
            float(observation_time),
            self.scenario,
            self.collect_stats,
            self.k_cutoff,
            self.with_self_ctmn,
        )
        return cast(JuliaSimulationResult, result)

    def __call__(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        if self.scenario == "constant_period":
            return self.__call_cst_period__(observation_time, seed)

        if self.use_julia:
            return self.__call_julia__(observation_time, seed)

        reset_mode = self.scenario == "reset_interval"
        T = observation_time
        event_arrivals, ctmn_arrivals, ctmn_periods = self._generate_data(
            observation_time, seed
        )
        ctmn_intervals_union = DisjointUnion([])

        for arrival, period in zip(ctmn_arrivals, ctmn_periods):
            ctmn_intervals_union.add_interval(
                Interval.from_start(arrival, period).capped(T), reset_mode=reset_mode
            )
        if self.k_cutoff:
            for interval in ctmn_intervals_union.intervals:
                num_arrivals = np.sum(
                    (ctmn_arrivals >= interval.start) & (ctmn_arrivals < interval.stop)
                )
                if self.k_cutoff and num_arrivals > self.k_cutoff:
                    ctmn_intervals_union.intervals.remove(interval)
        ctmn_length = ctmn_intervals_union.length
        contaminated_events = (
            np.array([t for t in event_arrivals if ctmn_intervals_union.contains(t)])
            if self.collect_events
            else None
        )
        # Classify the contamination intervals per number of contamination arrivals in the interval
        ctmn_int_categories: dict[int, int] = {}
        if self.collect_stats:
            for interval in ctmn_intervals_union.intervals:
                num_arrivals = np.sum(
                    (ctmn_arrivals >= interval.start) & (ctmn_arrivals < interval.stop)
                )
                ctmn_int_categories[num_arrivals] = (
                    ctmn_int_categories.get(num_arrivals, 0) + 1
                )

        if not self.with_self_ctmn:
            result = SimulationResult(
                event_arrivals,
                ctmn_arrivals,
                ctmn_periods,
                ctmn_intervals_union,
                ctmn_length,
                contaminated_events,
                ctmn_int_categories,
                None,
            )

        else:
            ctmn_hierarchy: dict[
                int, dict[Literal["culprits", "victims"], npt.NDArray[np.int_]]
            ] = {}
            for n_skip in range(1, len(ctmn_arrivals)):
                skip_diffs = ctmn_arrivals[n_skip:] - ctmn_arrivals[:-n_skip]
                mask = skip_diffs <= ctmn_periods[:-n_skip]
                # Indices of the culprit and victim events
                culprits = np.where(mask)[0]
                victims = n_skip + culprits
                ctmn_hierarchy[n_skip] = {
                    "culprits": culprits,
                    "victims": victims,
                }

            def get_idx(role: Literal["culprits", "victims"]):
                idx = np.unique(
                    np.concatenate(
                        [
                            ctmn_hierarchy[n_skip][role]
                            for n_skip in range(1, len(ctmn_arrivals))
                        ]
                    )
                )
                return idx

            try:
                victims_ = ctmn_arrivals[get_idx("victims")]
                culprits_ = ctmn_arrivals[get_idx("culprits")]
            except ValueError:
                # No victims or culprits found
                victims_ = np.array([])
                culprits_ = np.array([])

            self_ctmn_results = SelfCtmnSimulationResult(
                ctmn_hierarchy,
                culprits_,
                victims_,
            )

            result = SimulationResult(
                event_arrivals,
                ctmn_arrivals,
                ctmn_periods,
                ctmn_intervals_union,
                ctmn_length,
                contaminated_events,
                ctmn_int_categories,
                self_ctmn_results,
            )

        return result
