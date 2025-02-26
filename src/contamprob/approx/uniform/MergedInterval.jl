module MergedInterval

include("../Template.jl")
using .Template
using Integrals
export ProbByHand, mean, variance

@eval $(define_uniform_prob_struct(:ProbByHand))
@eval $(define_uniform_prob_struct(:Prob))


heaviside(x::Real) = x < 0 ? 0.0 : 1.0

function powstep(x::Real, n::Int)
	return x^n * heaviside(x)
end

function (prob::ProbByHand)(::Val{1}, T::Float64)
	tau_max = prob.tau_max
	lambda_T = prob.lambda * T
	return (1 / tau_max) * exp(-lambda_T) * heaviside(tau_max - T)
end

function (prob::ProbByHand)(::Val{2}, T::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_T = prob.lambda * T
	term1 = lambda * exp(-lambda_T) * (T^2 / (2 * tau_max^2)) * heaviside(tau_max - T)
	term2 = term1 + lambda * exp(-lambda_T) / tau_max^2 *
					(T - 2 * tau_max)^2 / 2 * heaviside(2 * tau_max - T) *
					heaviside(T - tau_max)
	return term1 + term2
end

function (prob::ProbByHand)(::Val{2}, T::Float64, tau::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_T = prob.lambda * T
	return 1 / tau_max^2 * lambda * exp(-lambda_T) * powstep(tau + tau_max - T, 1) * heaviside(tau_max - tau)
end

# function (prob::ProbByHand)(::Val{3}, T::Float64, tau::Float64)
# 	tau_max = prob.tau_max
# 	lambda = prob.lambda
# 	lambda_T = prob.lambda * T
# 	factor = lambda^2 * tau_max^(-3) * exp(-lambda_T)
# 	terms =
# 		T .* tau .* (powstep(-T + 2 * tau_max, 1) .* powstep(T - tau - tau_max, 0) + powstep(-tau + tau_max, 1) .* powstep(-T + tau + tau_max, 0)) - T .* tau .* powstep(-T + tau_max, 1) +
# 		T .* (powstep(tau_max, 2) .* powstep(-T + tau_max, 0) + powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0)) / 2 -
# 		T .* (powstep(tau_max, 2) .* powstep(-T + tau + tau_max, 0) + powstep(-T + tau + 2 * tau_max, 2) .* powstep(T - tau - tau_max, 0)) / 2 -
# 		T .* (powstep(-T + 2 * tau_max, 2) .* powstep(tau - tau_max, 0) + powstep(-tau + tau_max, 0) .* powstep(-T + tau + tau_max, 2)) / 2 + T .* powstep(-T + tau + tau_max, 2) / 2 -
# 		2 * tau .* tau_max .* (powstep(-T + 2 * tau_max, 1) .* powstep(T - tau - tau_max, 0) + powstep(-tau + tau_max, 1) .* powstep(-T + tau + tau_max, 0)) + 2 * tau .* tau_max .* powstep(-T + tau_max, 1) +
# 		tau .* (powstep(tau_max, 2) .* powstep(-T + tau + tau_max, 0) + powstep(-T + tau + 2 * tau_max, 2) .* powstep(T - tau - tau_max, 0)) / 2 +
# 		tau .* (powstep(-T + 2 * tau_max, 2) .* powstep(T - tau - tau_max, 0) + powstep(-tau + tau_max, 2) .* powstep(-T + tau + tau_max, 0)) / 2 - tau .* powstep(-T + tau_max, 2) / 2 - tau .* powstep(-T + tau + tau_max, 2) / 2 -
# 		tau_max .* (powstep(tau_max, 2) .* powstep(-T + tau_max, 0) + powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0)) +
# 		tau_max .* (powstep(tau_max, 2) .* powstep(-T + tau + tau_max, 0) + powstep(-T + tau + 2 * tau_max, 2) .* powstep(T - tau - tau_max, 0)) +
# 		tau_max .* (powstep(-T + 2 * tau_max, 2) .* powstep(tau - tau_max, 0) + powstep(-tau + tau_max, 0) .* powstep(-T + tau + tau_max, 2)) - tau_max .* powstep(-T + tau + tau_max, 2) + powstep(tau_max, 3) .* powstep(-T + tau_max, 0) / 3 -
# 		powstep(tau_max, 3) .* powstep(-T + tau + tau_max, 0) / 3 + powstep(-T + 2 * tau_max, 3) .* powstep(T - tau_max, 0) / 3 - powstep(-T + 2 * tau_max, 3) .* powstep(tau - tau_max, 0) / 3 -
# 		powstep(-tau + tau_max, 0) .* powstep(-T + tau + tau_max, 3) / 3 + powstep(-T + tau + tau_max, 3) / 3 - powstep(-T + tau + 2 * tau_max, 3) .* powstep(T - tau - tau_max, 0) / 3
# 	return factor * terms
# end



function (prob::ProbByHand)(::Val{3}, T::Float64)
	tau_max = prob.tau_max
	lambda = prob.lambda
	lambda_T = prob.lambda * T
	factor = lambda^2 * tau_max^(-3) * exp(-lambda_T)
	terms =
		T .^ 2 .* powstep(tau_max, 2) .* powstep(-T + tau_max, 0) / 2 + T .^ 2 .* powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0) / 2 - T .* tau_max .* powstep(tau_max, 2) .* powstep(-T + tau_max, 0) -
		T .* tau_max .* powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0) - T .* (T - powstep(T - tau_max, 1)) .^ 2 .* powstep(-T + tau_max, 1) / 2 +
		T .* (-(T - tau_max - powstep(T - tau_max, 1)) .^ 2 / 2 + (T - tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1)) .^ 2 / 2) .* powstep(-T + 2 * tau_max, 1) -
		T .* (-powstep(-tau_max, 1) + powstep(T - tau_max - powstep(T - tau_max, 1), 1)) .* powstep(-T + 2 * tau_max, 2) / 2 - T .* (-powstep(-T + tau_max, 1) + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* powstep(tau_max, 2) / 2 +
		T .* (
			-tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 / 2 + tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 / 2 +
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 - (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3
		) +
		T .* (
			-(-T + tau_max) .* (T - tau_max + powstep(-T + tau_max, 1)) .^ 2 + (-T + tau_max) .* (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 - (T - tau_max + powstep(-T + tau_max, 1)) .^ 3 / 3 -
			(T - tau_max + powstep(-T + tau_max, 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2) + (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 +
			(T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2)
		) / 2 -
		T .* (
			-(-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 + (-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 -
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 - (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2) +
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 +
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2)
		) / 2 -
		T .* (
			-(-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 + (-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 -
			(T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 - (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (T .^ 2 - 4 * T .* tau_max + 4 * tau_max .^ 2) +
			(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 + (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .* (T .^ 2 - 4 * T .* tau_max + 4 * tau_max .^ 2)
		) / 2 - T .* powstep(tau_max, 2) .* powstep(-T + tau_max, 0) .* powstep(T - tau_max, 1) / 2 + T .* powstep(tau_max, 3) .* powstep(-T + tau_max, 0) / 3 - T .* powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0) .* powstep(T - tau_max, 1) / 2 +
		T .* powstep(-T + 2 * tau_max, 3) .* powstep(T - tau_max, 0) / 3 - tau_max .^ 2 .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 / 4 +
		tau_max .^ 2 .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 / 4 + tau_max .* (T - powstep(T - tau_max, 1)) .^ 2 .* powstep(-T + tau_max, 1) -
		2 * tau_max .* (-(T - tau_max - powstep(T - tau_max, 1)) .^ 2 / 2 + (T - tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1)) .^ 2 / 2) .* powstep(-T + 2 * tau_max, 1) +
		tau_max .* (-powstep(-tau_max, 1) + powstep(T - tau_max - powstep(T - tau_max, 1), 1)) .* powstep(-T + 2 * tau_max, 2) + tau_max .* (-powstep(-T + tau_max, 1) + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* powstep(tau_max, 2) +
		tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 - tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 +
		2 * tau_max .* (
			tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 / 2 - tau_max .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 / 2 -
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 + (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3
		) -
		tau_max .* (
			-(-T + tau_max) .* (T - tau_max + powstep(-T + tau_max, 1)) .^ 2 + (-T + tau_max) .* (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 - (T - tau_max + powstep(-T + tau_max, 1)) .^ 3 / 3 -
			(T - tau_max + powstep(-T + tau_max, 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2) + (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 +
			(T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2)
		) +
		tau_max .* (
			-(-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 + (-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 -
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 - (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2) +
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 +
			(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .* (T .^ 2 - 2 * T .* tau_max + tau_max .^ 2)
		) +
		tau_max .* (
			-(-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 + (-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 -
			(T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 - (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (T .^ 2 - 4 * T .* tau_max + 4 * tau_max .^ 2) +
			(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 + (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .* (T .^ 2 - 4 * T .* tau_max + 4 * tau_max .^ 2)
		) + tau_max .* powstep(tau_max, 2) .* powstep(-T + tau_max, 0) .* powstep(T - tau_max, 1) + tau_max .* powstep(-T + 2 * tau_max, 2) .* powstep(T - tau_max, 0) .* powstep(T - tau_max, 1) -
		(-T + tau_max) .* (T - tau_max + powstep(-T + tau_max, 1)) .^ 3 / 3 + (-T + tau_max) .* (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 +
		(-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 3 / 3 - (-T + tau_max) .* (T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 +
		(-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 3 - (-T + 2 * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 3 +
		(-2 // 3 * T + (2 // 3) * tau_max) .* (T - tau_max + powstep(-T + tau_max, 1)) .^ 3 / 2 - (-2 // 3 * T + (2 // 3) * tau_max) .* (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 2 -
		(-2 // 3 * T + (4 // 3) * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 3 / 2 +
		(-2 // 3 * T + (4 // 3) * tau_max) .* (T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 3 / 2 - (T - powstep(T - tau_max, 1)) .^ 2 .* powstep(-T + tau_max, 2) / 4 +
		(-(T - tau_max + powstep(-T + tau_max, 1)) .^ 2 / 2 + (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 / 2) .* powstep(tau_max, 2) / 2 +
		(-(T - tau_max - powstep(T - tau_max, 1)) .^ 2 / 2 + (T - tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1)) .^ 2 / 2) .* powstep(-T + 2 * tau_max, 2) / 2 -
		(-powstep(-tau_max, 1) + powstep(T - tau_max - powstep(T - tau_max, 1), 1)) .* powstep(-T + 2 * tau_max, 3) / 3 - (-powstep(-T + tau_max, 1) + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* powstep(tau_max, 3) / 3 -
		(T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 4 / 24 - (T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 .* (T .^ 2 / 2 - 2 * T .* tau_max + 2 * tau_max .^ 2) / 2 +
		(T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 6 * T .* tau_max + 6 * tau_max .^ 2) / 3 +
		(T - 2 * tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (-T .^ 3 + 6 * T .^ 2 .* tau_max - 12 * T .* tau_max .^ 2 + 8 * tau_max .^ 3) / 3 +
		(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 4 / 24 +
		(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 .* (T .^ 2 / 2 - 2 * T .* tau_max + 2 * tau_max .^ 2) / 2 -
		(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 6 * T .* tau_max + 6 * tau_max .^ 2) / 3 -
		(T - 2 * tau_max + powstep(tau_max - powstep(-tau_max + powstep(T - tau_max, 1), 1), 1)) .* (-T .^ 3 + 6 * T .^ 2 .* tau_max - 12 * T .* tau_max .^ 2 + 8 * tau_max .^ 3) / 3 + (T - tau_max + powstep(-T + tau_max, 1)) .^ 4 / 24 +
		(T - tau_max + powstep(-T + tau_max, 1)) .^ 2 .* (T .^ 2 / 2 - T .* tau_max + tau_max .^ 2 / 2) / 2 - (T - tau_max + powstep(-T + tau_max, 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 3 * T .* tau_max + (3 // 2) * tau_max .^ 2) / 3 -
		(T - tau_max + powstep(-T + tau_max, 1)) .* (-T .^ 3 + 3 * T .^ 2 .* tau_max - 3 * T .* tau_max .^ 2 + tau_max .^ 3) / 3 - (T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 4 / 24 -
		(T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 .* (T .^ 2 / 2 - T .* tau_max + tau_max .^ 2 / 2) / 2 +
		(T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 3 * T .* tau_max + (3 // 2) * tau_max .^ 2) / 3 +
		(T - tau_max + powstep(tau_max - powstep(T - tau_max, 1), 1)) .* (-T .^ 3 + 3 * T .^ 2 .* tau_max - 3 * T .* tau_max .^ 2 + tau_max .^ 3) / 3 - (T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 4 / 24 +
		(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 3 * T .* tau_max + (3 // 2) * tau_max .^ 2) / 3 +
		(T - tau_max + powstep(-T + 2 * tau_max - powstep(tau_max, 1), 1)) .* (-T .^ 3 + 3 * T .^ 2 .* tau_max - 3 * T .* tau_max .^ 2 + tau_max .^ 3) / 3 +
		(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 4 / 24 -
		(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .^ 2 .* ((3 // 2) * T .^ 2 - 3 * T .* tau_max + (3 // 2) * tau_max .^ 2) / 3 -
		(T - tau_max + powstep(-T + 2 * tau_max - powstep(-T + tau_max + powstep(T - tau_max, 1), 1), 1)) .* (-T .^ 3 + 3 * T .^ 2 .* tau_max - 3 * T .* tau_max .^ 2 + tau_max .^ 3) / 3 -
		powstep(tau_max, 3) .* powstep(-T + tau_max, 0) .* powstep(T - tau_max, 1) / 3 - powstep(-T + 2 * tau_max, 3) .* powstep(T - tau_max, 0) .* powstep(T - tau_max, 1) / 3
	return factor * terms
end


end # module
