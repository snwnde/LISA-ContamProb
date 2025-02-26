module MergedInterval

include("../Template.jl")
using .Template
export ProbByHand, mean, variance

@eval $(define_exp_prob_struct(:ProbByHand))
@eval $(define_exp_prob_struct(:Prob))

function (prob::ProbByHand)(::Val{1}, T::Float64)
	return prob.nu * exp(-(prob.lambda + prob.nu) * T)
end

function (prob::ProbByHand)(::Val{2}, T::Float64)
	return 2 * prob.lambda * prob(1, T) * (T - (1 / prob.nu) * (1 - exp(-prob.nu * T)))
end

function (prob::ProbByHand)(::Val{3}, T::Float64)
	nu_T = prob.nu * T
	nu = prob.nu
	nu_2 = prob.nu^2
	return 2 * prob.lambda^2 * prob(1, T) *
		   (
			   T^2
			   +
			   T / nu * (exp(-nu_T) - 5 / 2)
			   -
			   3 / nu_2 * exp(-nu_T)
			   + 3 / 4 / nu_2 * exp(-2 * nu_T)
			   + 9 / 4 / nu_2
		   )
end

end # module
