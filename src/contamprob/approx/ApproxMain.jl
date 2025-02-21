module ApproxMain

module Exponential
	include("exponential/MergedInterval.jl")
	include("exponential/ResetInterval.jl")
	import .MergedInterval
	import .ResetInterval
end

module Uniform
	include("uniform/MergedInterval.jl")
	include("uniform/ResetInterval.jl")
	import .MergedInterval
	import .ResetInterval
end

end # module
