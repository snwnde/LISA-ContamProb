module SimuCore

using PythonCall
export ctmn_simulate

struct Interval
	start::Float64
	stop::Float64

	function Interval(start::Float64, stop::Float64, do_check::Bool = false)
		if do_check
			@assert start <= stop "The start time must be less than or equal to the stop time."
		end
		return new(start, stop)
	end

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

function are_disjoint(intervals::Vector{Interval})
	for i in 1:length(intervals)-1
		for j in i+1:length(intervals)
			if !are_disjoint(intervals[i], intervals[j])
				return false
			end
		end
	end
	return true
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

	function DisjointUnion(intervals::Vector{Interval}, do_check::Bool = false)
		if do_check
			@assert are_disjoint(intervals) "The intervals must be disjoint."
		end
		return new(intervals)
	end
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
	if isempty(union.intervals)
		return 0.0
	end
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
	ctmn_int_categories::Dict{Int, Int}
end

function ctmn_simulate(
	event_arrivals::Union{Nothing, PyArray{Float64, 1, true, true, Float64}},
	ctmn_arrivals::PyArray{Float64, 1, true, true, Float64},
	ctmn_periods::PyArray{Float64, 1, true, true, Float64},
	observation_time::Float64, scenario::String, collect_stats::Bool, k_cutoff::Union{Int, Nothing})
	reset_mode = scenario == "reset_interval"
	T = observation_time
	ctmn_intervals_union = DisjointUnion(Vector{Interval}())
	for (arrival, period) in zip(ctmn_arrivals, ctmn_periods)
		ctmn_intervals_union = add_interval(ctmn_intervals_union, cap(Interval(arrival, arrival + period), T), reset_mode)
	end
	if k_cutoff !== nothing
		ctmn_intervals_union = DisjointUnion([interval for interval in ctmn_intervals_union.intervals if sum((ctmn_arrivals .>= interval.start) .& (ctmn_arrivals .< interval.stop)) <= k_cutoff])
	end
	ctmn_length = measure(ctmn_intervals_union)
	contaminated_events = event_arrivals !== nothing ? [t for t in event_arrivals if contains_element(ctmn_intervals_union, t)] : nothing
	# Classify the contamination intervals per number of contamination arrivals in the interval
	ctmn_int_categories = Dict{Int, Int}()
	if collect_stats
		for interval in ctmn_intervals_union.intervals
			num_arrivals = sum((ctmn_arrivals .>= interval.start) .& (ctmn_arrivals .< interval.stop))
			ctmn_int_categories[num_arrivals] = get(ctmn_int_categories, num_arrivals, 0) + 1
		end
	end
	return SimulationResult(event_arrivals, ctmn_arrivals, ctmn_periods, ctmn_intervals_union,
		ctmn_length, contaminated_events, ctmn_int_categories)
end

end # module
