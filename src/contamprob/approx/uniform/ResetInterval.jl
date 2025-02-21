module ResetInterval

include("../Template.jl")
using .Template
import .Template: k_weighted
using Integrals
export ProbByHand, mean, variance, k_weighted

@eval $(define_uniform_prob_struct(:ProbByHand))
@eval $(define_uniform_prob_struct(:Prob))

heaviside(x::Real) = x < 0 ? 0.0 : 1.0

function (prob::ProbByHand)(::Val{1}, t::Float64)
	tau_max = prob.tau_max
	lambda_t = prob.lambda * t
	return exp(-lambda_t) / tau_max * heaviside(tau_max - t)
end

function (prob::ProbByHand)(::Val{2}, t::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_t = prob.lambda * t
	factor = lambda * exp(-lambda_t) / tau_max^2
	case1 = heaviside(tau_max - t)
	case2 = heaviside(2 * tau_max - t) * heaviside(t - tau_max)
	term1 = factor * (t * (tau_max - t / 2)) * case1
	term2 = factor * (2 * tau_max^2 - 2 * t * tau_max + 1 / 2 * t^2) * case2
	return term1 + term2
end

# This is wrong but for now it is not used
function (prob::ProbByHand)(::Val{3}, t::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_t = prob.lambda * t
	factor = lambda^2 * exp(-lambda_t) / tau_max^3
	case1 = heaviside(tau_max - t)
	case2 = heaviside(2 * tau_max - t) * heaviside(t - tau_max)
	case3 = heaviside(3 * tau_max - t) * heaviside(t - 2 * tau_max)
	term1 = factor * (t^4 / 24 - tau_max * t^3 / 3 + tau_max^2 * t^2 / 2) * case1
	term2 = factor * ((2 * tau_max^4 - t^4 - (t - tau_max)^4) / 24 + tau_max / 3 * (t^3 + (t - tau_max)^3 - 2 * tau_max^3)
					  + tau_max^2 / 2 * (3 * tau_max^2 - 2 * t^2 - (t - tau_max)^2) + 4 * tau_max^3 / 3 * (t - tau_max)) * case2
	term3 = factor * (((t - tau_max)^4 - 16 * tau_max^4) / 24 - tau_max / 3 * ((t - tau_max)^3 + 8 * tau_max^3)
					  + tau_max^2 * ((t - tau_max)^2 - 4 * tau_max^2) + 4 * tau_max^3 / 3 * (3 * tau_max - t)) * case3
	return term1 + term2 + term3
end

end # module
