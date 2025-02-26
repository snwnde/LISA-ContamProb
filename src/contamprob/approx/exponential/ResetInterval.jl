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

function (prob::ProbByHand)(T::Float64)
	return prob.nu * exp(-prob.nu * T)
end

function k_weighted(prob::ProbByHand, T::Float64)
	lambda = prob.lambda
	nu = prob.nu
	nu_T = prob.nu * T
	return (lambda * nu_T + nu) * exp(-nu_T)
end

end # module
