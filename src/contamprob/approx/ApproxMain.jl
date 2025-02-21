module ApproxMain

module Exponential
	include("exponential/MergedInterval.jl")
	include("exponential/ResetInterval.jl")
	import .MergedInterval
	import .ResetInterval
end

module Uniform
	include("uniform/MergedInterval.jl")
	import .MergedInterval
end

end # module
