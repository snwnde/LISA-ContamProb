module ResetInterval

include("../Template.jl")
using .Template
import .Template: k_weighted
using Integrals
export ProbByHand, mean, variance, k_weighted

@eval $(define_exp_prob_struct(:ProbByHand))

# Make Prob an alias of ProbByHand 
# since in this case we can compute all orders by hand
const Prob = ProbByHand

function (prob::ProbByHand)(t::Float64)
	return prob.nu * exp(-prob.nu * t)
end

function k_weighted(prob::ProbByHand, t::Float64)
	lambda = prob.lambda
	nu = prob.nu
	nu_t = prob.nu * t
	return (lambda * nu_t + nu) * exp(-nu_t)
end

end # module
