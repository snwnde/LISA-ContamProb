module SimuCore

using PythonCall
export result_gen

struct Interval
	start::Float64
	stop::Float64
end

function isvalid(interval::Interval)
	return interval.start <= interval.stop
end

function measure(interval::Interval)
	return interval.stop - interval.start
end

function cap(interval::Interval, capby::Float64)
	return Interval(interval.start, min(interval.stop, capby))
end

function contains_element(interval::Interval, element::Float64)
	return interval.start <= element <= interval.stop
end

function are_disjoint(interval::Interval, other::Interval)
	return interval.stop < other.start || other.stop < interval.start
end

function merge_intervals(interval::Interval, other::Interval, reset_mode::Bool = false)
	@assert !are_disjoint(interval, other) "The intervals must not be disjoint."
	if !reset_mode
		return Interval(min(interval.start, other.start), max(interval.stop, other.stop))
	end
	if interval.start <= other.start
		return Interval(interval.start, other.stop)
	end
	return Interval(other.start, interval.stop)
end

struct DisjointUnion
	intervals::Vector{Interval}
end

function add_interval(union::DisjointUnion, other::Interval, reset_mode::Bool = false)
	intervals = copy(union.intervals)
	idx = searchsortedfirst(intervals, other, by = interval -> interval.start)
	if idx > 1 && !are_disjoint(intervals[idx-1], other)
		idx -= 1
	end
	while idx <= length(intervals) && !are_disjoint(intervals[idx], other)
		other = merge_intervals(other, intervals[idx], reset_mode)
		deleteat!(intervals, idx)
	end
	insert!(intervals, idx, other)
	return DisjointUnion(intervals)
end

function measure(union::DisjointUnion)
	return sum(measure(interval) for interval in union.intervals)
end

function contains_element(union::DisjointUnion, element::Float64)
	return any(contains_element(interval, element) for interval in union.intervals)
end

struct SimulationResult
	event_arrivals::Union{Nothing, PyArray{Float64, 1, true, true, Float64}}
	ctmn_arrivals::PyArray{Float64, 1, true, true, Float64}
	ctmn_periods::PyArray{Float64, 1, true, true, Float64}
	ctmn_intervals::DisjointUnion
	ctmn_length::Float64
	contaminated_events::Union{Nothing, PyArray{Float64, 1, true, true, Float64}}
end

function result_gen(
	event_arrivals::Union{Nothing, PyArray{Float64, 1, true, true, Float64}},
	ctmn_arrivals::PyArray{Float64, 1, true, true, Float64},
	ctmn_periods::PyArray{Float64, 1, true, true, Float64},
	observation_time::Float64, scenario::String)
	reset_mode = scenario == "reset_interval"
	T = observation_time
	ctmn_intervals_union = DisjointUnion(Vector{Interval}())
	for (arrival, period) in zip(ctmn_arrivals, ctmn_periods)
		ctmn_intervals_union = add_interval(ctmn_intervals_union, cap(Interval(arrival, arrival + period), T), reset_mode)
	end
	ctmn_length = measure(ctmn_intervals_union)
	contaminated_events = event_arrivals !== nothing ? [t for t in event_arrivals if contains_element(ctmn_intervals_union, t)] : nothing
	return SimulationResult(event_arrivals, ctmn_arrivals, ctmn_periods, ctmn_intervals_union, ctmn_length, contaminated_events)
end

end # module
