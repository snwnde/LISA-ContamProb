module Template

using Integrals

export define_exp_prob_struct, define_uniform_prob_struct, define_prob_methods

function define_exp_prob_struct(name::Symbol)
	return quote
		struct $name
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
		struct $name
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


function define_prob_methods(name::Symbol)
	return quote
		function (prob::$name)(k::Int, t::Float64)
			return prob(Val(k), t)
		end

		function (prob::$name)(t::Float64)
			return sum([prob(k, t) for k in 1:prob.max_k])
		end

		function mean(prob::$name, obs_time::Float64 = Inf)
			domain = (0, obs_time)
			int = IntegralProblem((x, _) -> x * prob(x), domain)
			sol = solve(int, HCubatureJL())
			return sol[1]
		end

		function variance(prob::$name, obs_time::Float64 = Inf)
			domain = (0, obs_time)
			mean_val = mean(prob)
			int = IntegralProblem((x, _) -> (x - mean_val)^2 * prob(x), domain)
			sol = solve(int, HCubatureJL())
			return sol[1]
		end
	end
end

end # module
