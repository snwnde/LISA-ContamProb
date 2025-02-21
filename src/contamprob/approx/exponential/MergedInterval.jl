module MergedInterval

include("../Template.jl")
using .Template
export ProbByHand, mean, variance

@eval $(define_exp_prob_struct(:ProbByHand))
@eval $(define_exp_prob_struct(:Prob))

function (prob::ProbByHand)(::Val{1}, t::Float64)
	return prob.nu * exp(-(prob.lambda + prob.nu) * t)
end

function (prob::ProbByHand)(::Val{2}, t::Float64)
	return 2 * prob.lambda * prob(1, t) * (t - (1 / prob.nu) * (1 - exp(-prob.nu * t)))
end

function (prob::ProbByHand)(::Val{3}, t::Float64)
	nu_t = prob.nu * t
	nu = prob.nu
	nu_2 = prob.nu^2
	return 2 * prob.lambda^2 * prob(1, t) *
		   (
			   t^2
			   +
			   t / nu * (exp(-nu_t) - 5 / 2)
			   -
			   3 / nu_2 * exp(-nu_t)
			   + 3 / 4 / nu_2 * exp(-2 * nu_t)
			   + 9 / 4 / nu_2
		   )
end

end # module
