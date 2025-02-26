module ResetInterval

include("../Template.jl")
using .Template
import .Template: k_weighted
using Integrals
export ProbByHand, mean, variance, k_weighted

@eval $(define_uniform_prob_struct(:ProbByHand))
@eval $(define_uniform_prob_struct(:Prob))

heaviside(x::Real) = x < 0 ? 0.0 : 1.0

function (prob::ProbByHand)(::Val{1}, T::Float64)
	tau_max = prob.tau_max
	lambda_T = prob.lambda * T
	return exp(-lambda_T) / tau_max * heaviside(tau_max - T)
end

function (prob::ProbByHand)(::Val{2}, T::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_T = prob.lambda * T
	factor = lambda * exp(-lambda_T) / tau_max^2
	case1 = heaviside(tau_max - T)
	case2 = heaviside(2 * tau_max - T) * heaviside(T - tau_max)
	term1 = factor * (T * (tau_max - T / 2)) * case1
	term2 = factor * (2 * tau_max^2 - 2 * T * tau_max + 1 / 2 * T^2) * case2
	return term1 + term2
end

# This is wrong but for now it is not used
function (prob::ProbByHand)(::Val{3}, T::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_T = prob.lambda * T
	factor = lambda^2 * exp(-lambda_T) / tau_max^3
	case1 = heaviside(tau_max - T)
	case2 = heaviside(2 * tau_max - T) * heaviside(T - tau_max)
	case3 = heaviside(3 * tau_max - T) * heaviside(T - 2 * tau_max)
	term1 = factor * (T^4 / 24 - tau_max * T^3 / 3 + tau_max^2 * T^2 / 2) * case1
	term2 = factor * ((2 * tau_max^4 - T^4 - (T - tau_max)^4) / 24 + tau_max / 3 * (T^3 + (T - tau_max)^3 - 2 * tau_max^3)
					  + tau_max^2 / 2 * (3 * tau_max^2 - 2 * T^2 - (T - tau_max)^2) + 4 * tau_max^3 / 3 * (T - tau_max)) * case2
	term3 = factor * (((T - tau_max)^4 - 16 * tau_max^4) / 24 - tau_max / 3 * ((T - tau_max)^3 + 8 * tau_max^3)
					  + tau_max^2 * ((T - tau_max)^2 - 4 * tau_max^2) + 4 * tau_max^3 / 3 * (3 * tau_max - T)) * case3
	return term1 + term2 + term3
end

end # module
