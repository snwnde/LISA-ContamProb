module SingletonPopulation

export Problem

using Polynomials: Polynomials

δ(i::Int, j::Int)::Int = i == j ? 1 : 0

function Polynomial(coeffs::AbstractVector)
	"""Get a polynomial from its coefficients."""
	if isempty(coeffs)
		return Polynomials.Polynomial([0])
	end
	return Polynomials.Polynomial(coeffs)
end

function sum(gen::Union{Base.Generator, Base.Iterators.Flatten})
	if isempty(gen)
		return 0.0
	end
	return Base.sum(gen)
end

MainInd = @NamedTuple{n::Int, l::Int, m::Int, j::Int}
AuxInd = @NamedTuple{q::Int, r::Int, m::Int, j::Int}

mutable struct Table
	a::Dict{MainInd, Float64}
	A::Dict{MainInd, Float64}
	ξ::Dict{AuxInd, Float64}
	Ξ::Dict{AuxInd, Float64}
	ζ::Dict{AuxInd, Float64}
	Z::Dict{AuxInd, Float64}
	# Above not the greek letter "Z" but the latin letter "Z"
	η::Dict{AuxInd, Float64}
	θ::Dict{AuxInd, Float64}
	max_n::Int
	max_l::Int

	function Table(max_n::Int, max_l::Int, a = Dict(), A = Dict(), ξ = Dict(), Ξ = Dict(),
		ζ = Dict(), Z = Dict(), η = Dict(), θ = Dict())
		return new(a, A, ξ, Ξ, ζ, Z, η, θ, max_n, max_l)

	end
end

# Custom getindex methods for the dictionaries within Table
function Base.getindex(dict::Dict{MainInd, Float64}, ind::MainInd)
	if ind.m == -1 || ind.m == ind.n + 1
		return 0.0
	end
	return dict[ind]
end

function Base.getindex(dict::Dict{AuxInd, Float64}, ind::AuxInd)
	if ind.m == -1
		return 0.0
	end
	return dict[ind]
end

function Base.getindex(dict::Dict{MainInd, Float64}, ind::Tuple{Int, Int, Int, Int})
	return dict[(n = ind[1], l = ind[2], m = ind[3], j = ind[4])]
end

function Base.getindex(dict::Dict{AuxInd, Float64}, ind::Tuple{Int, Int, Int, Int})
	return dict[(q = ind[1], r = ind[2], m = ind[3], j = ind[4])]
end


struct Problem
	ctmn_rate::Float64
	event_rate::Float64
	ctmn_period::Float64
	λ::Float64
	μ::Float64
	τ::Float64
	table::Table

	function Problem(ctmn_rate::Float64, event_rate::Float64,
		ctmn_period::Float64, table::Table = Table(-1, -1))
		return new(ctmn_rate, event_rate, ctmn_period, ctmn_rate,
			event_rate, ctmn_period, table)
	end
end


function fill_ξ!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	ξ = problem.table.ξ
	if r > j
		ξ[ind] = 0.0
	elseif q == 0
		ξ[ind] = -factorial(j) / factorial(r) / (-μ)^(1 + j - r)
	elseif j == 0
		@assert r == 0 "r must be 0"
		ξ[ind] = 1.0 / μ
	else
		ξ[ind] = ξ[(q - 1, r, m, j)] - j / μ * ξ[(q, r, m, j - 1)]
	end

end


function fill_ζ!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	ζ = problem.table.ζ

	if r > j
		ζ[ind] = 0.0
	elseif q == 0
		ζ[ind] = exp(-μ * τ) * factorial(j) / factorial(r) / (-μ)^(1 + j - r)
	elseif j == 0
		@assert r == 0 "r must be 0"
		ζ[ind] = -exp(-μ * τ) * Polynomial([1 / factorial(k) for k in 0:q])(μ * τ) / μ
	else
		term_1 = (μ * τ)^q / factorial(q) * exp(-μ * τ) / (-μ) * δ(r, j)
		term_2_3 = ζ[(q - 1, r, m, j)] - j / μ * ζ[(q, r, m, j - 1)]
		ζ[ind] = term_1 + term_2_3
	end
end

function fill_Ξ!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	Ξ = problem.table.Ξ

	if r > q
		Ξ[ind] = 0.0
	elseif q == 0
		@assert r == 0 "r must be 0"
		Ξ[ind] = exp(μ * m * τ) * factorial(j) / (-μ)^(1 + j)
	elseif j == 0
		Ξ[ind] = -1 / factorial(r) * μ^(r - 1) * exp(μ * m * τ)
	else
		Ξ[ind] = Ξ[(q - 1, r, m, j)] - j / μ * Ξ[(q, r, m, j - 1)]
	end
end


function fill_Z!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	Z = problem.table.Z

	if r > q
		Z[ind] = 0.0
	elseif q == 0
		@assert r == 0 "r must be 0"
		Z[ind] = -exp(μ * m * τ) * factorial(j) / (-μ)^(1 + j)
	elseif j == 0
		Z[ind] = exp(μ * m * τ) * sum(
			1 / factorial(k) * μ^(k - 1) * binomial(k, r) * τ^(k - r) for k in r:q
		)
	else
		Z[ind] = Z[(q - 1, r, m, j)] - j / μ * Z[(q, r, m, j - 1)]
	end
end


function fill_η!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	η = problem.table.η

	if r > j + q + 1
		η[ind] = 0.0
	elseif q == 0
		if r == j + 1
			η[ind] = 1.0 / (j + 1)
		else
			η[ind] = 0.0
		end
	elseif j == 0
		if r == q + 1
			η[ind] = μ^q / factorial(q + 1)
		else
			η[ind] = 0.0
		end
	else
		η[ind] = μ / (j + 1) * η[(q - 1, r, m, j + 1)]
	end
end


function fill_θ!(problem::Problem, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	θ = problem.table.θ

	if r > j + q + 1
		θ[ind] = 0.0
	elseif q == 0
		if r == j + 1
			θ[ind] = -1.0 / (j + 1)
		else
			θ[ind] = 0.0
		end
	elseif j == 0
		if r > q + 1
			θ[ind] = 0.0
		else
			term_1 = μ^q / factorial(q + 1) * τ^(q + 1) * δ(r, 0)
			term_2 = -μ^q / factorial(q + 1) * binomial(q + 1, r) * τ^(q + 1 - r)
			θ[ind] = term_1 + term_2
		end
	else
		term_1 = -1.0 / (j + 1) * (μ * τ)^q / factorial(q) * δ(r, j + 1)
		term_2 = μ / (j + 1) * θ[(q - 1, r, m, j + 1)]
		θ[ind] = term_1 + term_2
	end
end

# function fill_a!(problem::Problem, ind::MainInd)
# 	μ = problem.μ
# 	τ = problem.τ
# 	n, l, m, j = ind.n, ind.l, ind.m, ind.j
# 	a = problem.table.a
# 	A = problem.table.A
# 	ξ = problem.table.ξ
# 	ζ = problem.table.ζ

# 	if n == 0
# 		a[ind] = δ(j, 0)
# 	elseif l == 0
# 		if j == 0
# 			term_1 = λ * exp(-μ * m * τ) * sum(
# 						 A[(n - 1, 0, m - 1, k)] * factorial(k) / (μ^(1 + k)) for k in 0:(n-1)
# 					 )
# 			term_2_3 = λ * sum(
# 				(-a[(n - 1, 0, m, k)] + a[(n - 1, 0, m - 1, k)] * exp(-μ * τ)) * factorial(k) / (-μ)^(1 + k) for k in 0:(n-1)
# 			)
# 			a[ind] = term_1 + term_2_3
# 		else
# 			term_1 = λ * exp(-μ * τ) * a[(n - 1, 0, m - 1, j - 1)] / j
# 			term_2_3 = λ * sum(
# 				(-a[(n - 1, 0, m, k)] + a[(n - 1, 0, m - 1, k)] * exp(-μ * τ)) * factorial(k) / factorial(j) / (-μ)^(1 + k - j) for k in j:(n-1)
# 			)
# 			a[ind] = term_1 + term_2_3
# 		end
# 	else
# 		if j == 0
# 			term_1 = λ * exp(-μ * m * τ) * Polynomial(
# 						 [1 / factorial(q) * sum(A[(n - 1, l - q, m - 1, k)] * factorial(k) / μ^(k + 1) for k in 0:(n+l-q)) for q in 0:l]
# 					 )(μ * τ)
# 			term_2_3 = λ * sum(
# 				sum(a[(n - 1, l - q, m, k)] * ξ[(q, 0, m, k)] + a[(n - 1, l - q, m - 1, k)] * ζ[(q, 0, m - 1, k)] for k in 0:(n+l-q)) for q in 0:l
# 			)
# 			a[ind] = term_1 + term_2_3
# 		else
# 			term_1 = λ * exp(-μ * τ) * Polynomial(
# 						 [a[(n - 1, l - q, m - 1, j - 1)] / j / factorial(q) for q in 0:min(l, n + l - j)]
# 					 )(μ * τ)
# 			term_2_3 = λ * sum(
# 				sum(a[(n - 1, l - q, m, k)] * ξ[(q, j, m, k)] + a[(n - 1, l - q, m - 1, k)] * ζ[(q, j, m - 1, k)] for k in j:(n+l-q)) for q in 0:l
# 			)
# 			a[ind] = term_1 + term_2_3
# 		end
# 	end
# end

function fill_a!(problem::Problem, ind::@NamedTuple{n::Int, l::Int, m::Int})
	μ = problem.μ
	λ = problem.λ
	τ = problem.τ
	n, l, m = ind.n, ind.l, ind.m
	a = problem.table.a
	A = problem.table.A
	ξ = problem.table.ξ
	ζ = problem.table.ζ

	if n == 0
		for j in 0:(n+l)
			a[(n = 0, l = l, m = 0, j = j)] = δ(j, 0)
		end
	elseif l == 0
		term_1 = λ * exp(-μ * m * τ) * sum(
					 A[(n - 1, 0, m - 1, k)] * factorial(k) / (μ^(1 + k)) for k in 0:(n-1)
				 )
		term_2_3 = λ * sum(
			(-a[(n - 1, 0, m, k)] + a[(n - 1, 0, m - 1, k)] * exp(-μ * τ)) * factorial(k) / (-μ)^(1 + k) for k in 0:(n-1)
		)
		a[(n = n, l = 0, m = m, j = 0)] = term_1 + term_2_3
		for j in 1:n
			term_1 = λ * exp(-μ * τ) * a[(n - 1, 0, m - 1, j - 1)] / j
			term_2_3 = λ * sum(
				(-a[(n - 1, 0, m, k)] + a[(n - 1, 0, m - 1, k)] * exp(-μ * τ)) * factorial(k) / factorial(j) / (-μ)^(1 + k - j) for k in j:(n-1)
			)
			a[(n = n, l = 0, m = m, j = j)] = term_1 + term_2_3
		end
	else
		term_1 = λ * exp(-μ * m * τ) * Polynomial(
					 [1 / factorial(q) * sum(A[(n - 1, l - q, m - 1, k)] * factorial(k) / μ^(k + 1) for k in 0:(n+l-q)) for q in 0:l]
				 )(μ * τ)
		term_2_3 = λ * sum(
			sum(a[(n - 1, l - q, m, k)] * ξ[(q, 0, m, k)] + a[(n - 1, l - q, m - 1, k)] * ζ[(q, 0, m - 1, k)] for k in 0:(n+l-q)) for q in 0:l
		)
		a[(n = n, l = l, m = m, j = 0)] = term_1 + term_2_3
		for j in 1:(n+l)
			term_1 = λ * exp(-μ * τ) * Polynomial(
						 [a[(n - 1, l - q, m - 1, j - 1)] / j / factorial(q) for q in 0:min(l, n + l - j)]
					 )(μ * τ)
			term_2_3 = λ * sum(
				sum(a[(n - 1, l - q, m, k)] * ξ[(q, j, m, k)] + a[(n - 1, l - q, m - 1, k)] * ζ[(q, j, m - 1, k)] for k in j:(n+l-q)) for q in 0:l
			)
			a[(n = n, l = l, m = m, j = j)] = term_1 + term_2_3
		end
	end
end


function fill_A!(problem::Problem, ind::@NamedTuple{n::Int, l::Int, m::Int})
	μ = problem.μ
	λ = problem.λ
	τ = problem.τ
	n, l, m = ind.n, ind.l, ind.m
	a = problem.table.a
	A = problem.table.A
	Ξ = problem.table.Ξ
	Z = problem.table.Z
	η = problem.table.η
	θ = problem.table.θ

	if n == 0
		for j in 0:(n+l)
			A[(n = 0, l = l, m = 0, j = j)] = 0.0
		end
	elseif l == 0
		term_1 = -λ * sum(
			A[(n - 1, 0, m - 1, k)] * factorial(k) / μ^(k + 1) for k in 0:(n-1)
		)
		term_2_3 = λ * exp(μ * m * τ) * sum(
					   (a[(n - 1, 0, m, k)] - a[(n - 1, 0, m - 1, k)] * exp(-μ * τ)) * factorial(k) / (-μ)^(k + 1) for k in 0:(n-1)
				   )
		A[(n = n, l = 0, m = m, j = 0)] = term_1 + term_2_3
		for j in 1:n
			term_1 = -λ * sum(
				A[(n - 1, 0, m - 1, k)] * factorial(k) / factorial(j) / μ^(k + 1 - j) for k in j:(n-1)
			)
			term_2_3 = λ / j * (A[(n - 1, 0, m, j - 1)] - A[(n - 1, 0, m - 1, j - 1)])
			A[(n = n, l = 0, m = m, j = j)] = term_1 + term_2_3
		end
	else
		for j in 0:(n+l)
			term_1 = -λ * Polynomial(
				[1 / factorial(q) * sum(A[(n - 1, l - q, m - 1, k)] * factorial(k) / factorial(j) / μ^(1 + k - j) for k in j:(n+l-q)) for q in 0:l]
			)(μ * τ)
			term_2_3 = λ * sum(
				sum(a[(n - 1, l - q, m, k)] * Ξ[(q, j, m, k)] + a[(n - 1, l - q, m - 1, k)] * Z[(q, j, m - 1, k)] for k in 0:(n+l-q)) for q in j:l
			)
			term_4_5 = λ * sum(
				sum(A[(n - 1, l - q, m, k)] * η[(q, j, m, k)] + A[(n - 1, l - q, m - 1, k)] * θ[(q, j, m - 1, k)] for k in max(j - q - 1, 0):(n+l-q)) for q in 0:l
			)
			A[(n = n, l = l, m = m, j = j)] = term_1 + term_2_3 + term_4_5
		end
	end
end


function fill!(problem::Problem, ind::@NamedTuple{n::Int, l::Int})
	max_n, max_l = ind.n, ind.l
	table = problem.table

	for n in (table.max_n+1):(max_n+1)
		if n == 0
			for l in (table.max_l+1):(max_l+1)
				for j in 0:(n+l)
					ind = (n = 0, l = l, m = 0, j = j)
					table.a[ind] = δ(j, 0)
					table.A[ind] = 0.0
				end
			end
		else
			for l in (table.max_l+1):(max_l+1)
				for m in 0:n
					for j in 0:(n+l)
						for r in 0:j
							ind = (q = 0, r = r, m = m, j = j)
							fill_ξ!(problem, ind)
							fill_ζ!(problem, ind)
							fill_Ξ!(problem, ind)
							fill_Z!(problem, ind)
						end
					end
					for j in 0:(n+l+1)
						for r in 0:(j+1)
							ind = (q = 0, r = r, m = m, j = j)
							fill_η!(problem, ind)
							fill_θ!(problem, ind)
						end
					end
					for q in 0:l
						for r in 0:q
							ind = (q = q, r = r, m = m, j = 0)
							fill_ξ!(problem, ind)
							fill_ζ!(problem, ind)
							fill_Ξ!(problem, ind)
							fill_Z!(problem, ind)
						end
						for r in 0:(q+1)
							ind = (q = q, r = r, m = m, j = 0)
							fill_η!(problem, ind)
							fill_θ!(problem, ind)
						end
					end
					for q in 1:l
						for j in 1:(n+l-q)
							for r in 0:(q+j)
								ind = (q = q, r = r, m = m, j = j)
								fill_ξ!(problem, ind)
								fill_ζ!(problem, ind)
								fill_Ξ!(problem, ind)
								fill_Z!(problem, ind)
							end
						end
						for j in 1:(n+l-q)
							for r in 0:(q+j+1)
								ind = (q = q, r = r, m = m, j = j)
								fill_η!(problem, ind)
								fill_θ!(problem, ind)
							end
						end
					end
					fill_a!(problem, (n = n, l = l, m = m))
					fill_A!(problem, (n = n, l = l, m = m))
				end
			end
		end
	end

	table.max_n = max_n
	table.max_l = max_l
end


function piecewise_monomial(problem::Problem, m::Int, j::Int)
	pow_step(t) = ifelse(t < 0, 0, t^j)
	return t -> pow_step(t - m * problem.τ)
end


function (problem::Problem)(max_n::Int, max_l::Int)
	"""Compute the P probability of the event at the given time."""
	fill!(problem, (n = max_n, l = max_l))

	function inner(observation_time)
		term_1 = exp(-problem.λ * observation_time) * sum(
			problem.table.a[(max_n, max_l, m, j)] *
			piecewise_monomial(problem, m, j)(observation_time)
			for m in 0:max_n
			for j in 0:(max_n+max_l)
		)
		term_2 =
			exp(-(problem.μ + problem.λ) * observation_time) * sum(
				problem.table.A[(max_n, max_l, m, j)] *
				piecewise_monomial(problem, m, j)(observation_time)
				for m in 0:max_n
				for j in 0:(max_n+max_l)
			)
		return term_1 + term_2
	end

	return inner
end

end # module
