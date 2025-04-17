module Template

using Integrals
using PythonCall

export BaseProb, define_exp_prob_struct, define_uniform_prob_struct, mean, variance, self_ctmn_num_mean,
	self_ctmn_num_variance, self_ctmn_covariance

abstract type BaseProb end

function define_exp_prob_struct(name::Symbol)
	return quote
		struct $name <: BaseProb
			ctmn_rate::Float64
			mean_ctmn::Float64
			lambda::Float64
			nu::Float64
			max_k::Int

			function $name(ctmn_rate::Float64, mean_ctmn::Float64, max_k::Int)
				return new(ctmn_rate, mean_ctmn, ctmn_rate, 1 / mean_ctmn, max_k)
			end
		end
	end
end

function define_uniform_prob_struct(name::Symbol)
	return quote
		struct $name <: BaseProb
			ctmn_rate::Float64
			max_ctmn::Float64
			lambda::Float64
			tau_max::Float64
			max_k::Int

			function $name(ctmn_rate::Float64, max_ctmn::Float64, max_k::Int)
				return new(ctmn_rate, max_ctmn, ctmn_rate, max_ctmn, max_k)
			end
		end
	end
end

function (prob::BaseProb)(k::Int, T::Float64, tau::Float64)
	return prob(Val(k), T, tau)
end

function (prob::BaseProb)(k::Int, T::Float64)
	try
		return prob(Val(k), T)
	catch e
		if isa(e, MethodError)
			domain = (0, T)
			int = IntegralProblem((tau, _) -> prob(k, T, tau), domain)
			sol = solve(int, HCubatureJL())
			return sol[1]
		else
			rethrow(e)
		end
	end
end

function p_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((x, _) -> prob(k, x), domain)
	sol = solve(int, HCubatureJL(); abstol = 1e-6)
	return sol[1]
end

function (prob::BaseProb)(T::Float64)
	return sum([prob(k, T) for k in 1:prob.max_k])
end

function (prob::BaseProb)(T::PyArray{Float64, 1, true, true, Float64})
	return [prob(T[i]) for i in eachindex(T)]
end

# function k_weighted(prob::BaseProb, T::Float64)
# 	return sum([k * prob(k, T) for k in 1:prob.max_k])
# end

function avg_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((T, _) -> k * prob(k, T), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end

# function avg_k(prob::BaseProb, obs_time::Float64 = Inf)
# 	domain = (0, obs_time)
# 	int = IntegralProblem((T, _) -> k_weighted(prob, T), domain)
# 	sol = solve(int, HCubatureJL())
# 	return sol[1]
# end

function var_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	avg_k_val = avg_k(prob, k, obs_time)
	domain = (0, obs_time)
	int = IntegralProblem((T, _) -> (k - avg_k_val)^2 * prob(k, T), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end

function M_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((x, _) -> x * prob(k, x), domain)
	sol = solve(int, HCubatureJL(); abstol = 1e-6)
	return sol[1]
end

function S_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((x, _) -> k * x * prob(k, x), domain)
	sol = solve(int, HCubatureJL(); abstol = 1e-6)
	return sol[1]
end

function mean(prob::BaseProb, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((x, _) -> x * prob(x), domain)
	sol = solve(int, HCubatureJL(); abstol = 1e-6)
	return sol[1]
end

function variance(prob::BaseProb, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	mean_val = mean(prob, obs_time)
	int = IntegralProblem((x, _) -> (x - mean_val)^2 * prob(x), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end


function aux_mean_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((T, _) -> (k-1) * prob(k, T), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end

function self_ctmn_num_mean(prob::BaseProb, obs_time::Float64 = Inf)
	return sum([aux_mean_k(prob, k, obs_time) for k in 2:prob.max_k])
end

function aux_var_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((T, _) -> (k-1-self_ctmn_num_mean(prob, obs_time))^2 * prob(k, T), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end

function aux_cov_k(prob::BaseProb, k::Int, obs_time::Float64 = Inf)
	domain = (0, obs_time)
	int = IntegralProblem((T, _) -> (k-1-self_ctmn_num_mean(prob, obs_time))*(T - mean(prob, obs_time)) * prob(k, T), domain)
	sol = solve(int, HCubatureJL())
	return sol[1]
end

function self_ctmn_num_variance(prob::BaseProb, obs_time::Float64 = Inf)
	return sum([aux_var_k(prob, k, obs_time) for k in 2:prob.max_k])
end

function self_ctmn_covariance(prob::BaseProb, obs_time::Float64 = Inf)
	return sum([aux_cov_k(prob, k, obs_time) for k in 2:prob.max_k])
end


# function self_ctmn_num_mean(prob::BaseProb, obs_time::Float64 = Inf)
# 	return sum([avg_k(prob, k, obs_time) for k in 2:prob.max_k])
# end

# function self_ctmn_num_variance(prob::BaseProb, obs_time::Float64 = Inf)
# 	return sum([
# 		var_k(prob, k, obs_time) + (2 - p_k(prob, k, obs_time)) * avg_k(prob, k, obs_time)^2 for k in 2:prob.max_k
# 	]) - self_ctmn_num_mean(prob, obs_time)^2
# end

# function self_ctmn_covariance(prob::BaseProb, obs_time::Float64 = Inf)
# 	return sum([
# 		S_k(prob, k, obs_time) - self_ctmn_num_mean(prob, obs_time) * M_k(prob, k, obs_time) -
# 		avg_k(prob, k, obs_time) * mean(prob, obs_time) for k in 2:prob.max_k
# 	]) + mean(prob, obs_time) * self_ctmn_num_mean(prob, obs_time)
# end

end # module
