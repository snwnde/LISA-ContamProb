"""Simulation of contamination probabilities."""

import bisect
from typing import NamedTuple, Self
import numpy as np
import numpy.typing as npt
from .problem_setup import ContaminationProcess, PoissonProcess


class Interval(NamedTuple):
    """An interval."""

    start: float
    """The start of the interval."""

    end: float
    """The end of the interval."""

    @classmethod
    def build(cls, start: float, end: float) -> Self:
        """Create an interval from a start and an end."""
        assert start <= end, (
            "The start of the interval must be less than or equal to the end."
        )
        return cls(start, end)

    @classmethod
    def from_start(cls, start: float, duration: float) -> Self:
        """Create an interval from a start and a duration."""
        return cls.build(start, start + duration)

    @classmethod
    def from_end(cls, end: float, duration: float) -> Self:
        """Create an interval from an end and a duration."""
        return cls.build(end - duration, end)

    @property
    def length(self) -> float:
        """Return the length of the interval."""
        return self.end - self.start

    def capped(self, cap: float) -> Self:
        """Cap the interval."""
        return type(self)(self.start, min(self.end, cap))

    def floored(self, floor: float) -> Self:
        """Floor the interval."""
        return type(self)(max(self.start, floor), self.end)

    def contains(self, element: float) -> bool:
        """Check if an element is contained in the interval."""
        return self.start <= element <= self.end

    def is_disjoint_with(self, other: Self) -> bool:
        """Check if two intervals are disjoint."""
        return self.end < other.start or other.end < self.start

    def merge_with(self, other: Self) -> Self:
        """Compute the union of two non-disjoint intervals."""
        assert not self.is_disjoint_with(other), "The intervals must not be disjoint."
        return type(self)(min(self.start, other.start), max(self.end, other.end))


class Union:
    """A union of intervals."""

    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = intervals

    def add_interval(self, other: Interval) -> Self:
        """Add an interval to the union and sort according to the start time."""
        return type(self)(self.intervals + [other])

    def contains(self, element: float) -> bool:
        """Check if an element is contained in the union."""
        return any(interval.contains(element) for interval in self.intervals)

    def sort(self):
        """Sort the intervals according to the start time."""
        self.intervals = sorted(self.intervals, key=lambda interval: interval.start)


class DisjointUnion(Union):
    """A union of disjoint intervals."""

    def add_interval(self, other: Interval) -> Self:
        """Add an interval to the union. If it overlaps with existing intervals, merge_with them.

        Caution: This method assumes that the intervals are sorted by start time.
        """
        # Copy the intervals
        intervals = self.intervals.copy()
        other_ = other
        # Find the position to insert the new interval
        idx = bisect.bisect_left(intervals, other_)
        # If the new interval overlaps with the previous interval, merge_with them
        if idx > 0 and not intervals[idx - 1].is_disjoint_with(other_):
            idx -= 1
            # We do not need to merge_with immediately since we will merge_with later

        # If the new interval overlaps with the next interval, merge_with them
        while idx < len(intervals) and not intervals[idx].is_disjoint_with(other_):
            other_ = other_.merge_with(intervals.pop(idx))
        # Insert the new (possibly merged) interval
        intervals.insert(idx, other_)
        return type(self)(intervals)

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


class SimulationResult(NamedTuple):
    """The result of a simulation."""

    event_arrivals: None | npt.NDArray[np.float_]
    """The arrival times of the events."""
    ctmn_arrivals: npt.NDArray[np.float_]
    """The arrival times of the contamination periods."""
    ctmn_periods: npt.NDArray[np.float_]
    """The durations of the contamination periods."""
    ctmn_intervals: DisjointUnion
    """The contamination intervals."""
    ctmn_length: float
    """The total length of the contamination intervals."""
    contaminated_events: None | npt.NDArray[np.float_]
    """The events that are contaminated."""


class Simulator:
    """Simulator for contamination probabilities."""

    def __init__(
        self,
        ctnm_proc: ContaminationProcess,
        event_proc: PoissonProcess,
        collect_events: bool = False,
    ) -> None:
        self.ctnm_proc = ctnm_proc
        self.event_proc = event_proc
        self.collect_events = collect_events

    def _generate_data(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        rngs = np.random.default_rng(seed).spawn(3)
        event_arrivals = (
            self.event_proc(observation_time, rngs[0]) if self.collect_events else None
        )
        ctmn_arrivals = self.ctnm_proc.process(observation_time, rngs[1])
        ctmn_periods = self.ctnm_proc.contamination(len(ctmn_arrivals), rngs[2])
        return event_arrivals, ctmn_arrivals, ctmn_periods

    def __call__(
        self, observation_time: float, seed: None | int | np.random.Generator = None
    ):
        T = observation_time
        event_arrivals, ctmn_arrivals, ctmn_periods = self._generate_data(
            observation_time, seed
        )
        ctmn_intervals_union = DisjointUnion([])

        for arrival, period in zip(ctmn_arrivals, ctmn_periods):
            ctmn_intervals_union = ctmn_intervals_union.add_interval(
                Interval.from_start(arrival, period).capped(T)
            )
        ctmn_length = ctmn_intervals_union.length
        contaminated_events = (
            np.array([t for t in event_arrivals if ctmn_intervals_union.contains(t)])
            if self.collect_events
            else None
        )
        result = SimulationResult(
            event_arrivals,
            ctmn_arrivals,
            ctmn_periods,
            ctmn_intervals_union,
            ctmn_length,
            contaminated_events,
        )
        return result
