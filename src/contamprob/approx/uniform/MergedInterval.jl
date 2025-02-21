module MergedInterval

include("../Template.jl")
using .Template
using Integrals
export ProbByHand, mean, variance

@eval $(define_uniform_prob_struct(:ProbByHand))
@eval $(define_uniform_prob_struct(:Prob))

@eval $(define_prob_methods(:ProbByHand))
@eval $(define_prob_methods(:Prob))


heaviside(x::Real) = x < 0 ? 0.0 : 1.0

function (prob::ProbByHand)(::Val{1}, t::Float64)
	tau_max = prob.tau_max
	lambda_t = prob.lambda * t
	return (1 / tau_max) * exp(-lambda_t) * heaviside(tau_max - t)
end

function (prob::ProbByHand)(::Val{2}, t::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_t = prob.lambda * t
	term1 = lambda * exp(-lambda_t) * (t^2 / (2 * tau_max^2)) * heaviside(tau_max - t)
	term2 = term1 + lambda * exp(-lambda_t) / tau_max^2 *
					(t - 2 * tau_max)^2 / 2 * heaviside(2 * tau_max - t) *
					heaviside(t - tau_max)
	return term1 + term2
end


end # module
