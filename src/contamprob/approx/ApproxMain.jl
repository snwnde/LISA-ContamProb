module ApproxMain

module Exponential
	include("exponential/MergedInterval.jl")
	import .MergedInterval
end

module Uniform
	include("uniform/MergedInterval.jl")
	import .MergedInterval
end

end # module
