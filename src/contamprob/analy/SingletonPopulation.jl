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

struct MainTable
	dict::Dict{MainInd, Float64}

	function MainTable()
		return new(Dict())
	end
end

struct AuxTable
	dict::Dict{AuxInd, Float64}

	function AuxTable()
		return new(Dict())
	end
end

function Base.setindex!(table::MainTable, value::Float64, ind::MainInd)
	table.dict[ind] = value
end

function Base.setindex!(table::AuxTable, value::Float64, ind::AuxInd)
	table.dict[ind] = value
end

function (table::MainTable)(n::Int, l::Int, m::Int, j::Int)
	ind = (n = n, l = l, m = m, j = j)
	if m == -1 || m == n + 1
		return 0.0
	end
	return table.dict[ind]
end

function (table::AuxTable)(q::Int, r::Int, m::Int, j::Int)
	ind = (q = q, r = r, m = m, j = j)
	if m == -1
		return 0.0
	end
	return table.dict[ind]
end


mutable struct AuxRegister
	ξ::AuxTable
	Ξ::AuxTable
	ζ::AuxTable
	Z::AuxTable # Not the greek letter "Z" but the latin letter "Z"
	η::AuxTable
	θ::AuxTable

	function AuxRegister(ξ = AuxTable(), Ξ = AuxTable(),
		ζ = AuxTable(), Z = AuxTable(), η = AuxTable(), θ = AuxTable())
		return new(ξ, Ξ, ζ, Z, η, θ)

	end
end


mutable struct Register
	a::MainTable
	A::MainTable
	aux::AuxRegister

	function Register(a = MainTable(), A = MainTable(),
		aux = AuxRegister())
		return new(a, A, aux)

	end
end


struct Problem
	ctmn_rate::Float64
	event_rate::Float64
	ctmn_period::Float64
	λ::Float64
	μ::Float64
	τ::Float64
	register::Register

	function Problem(ctmn_rate::Float64, event_rate::Float64,
		ctmn_period::Float64, register::Register = Register())
		return new(ctmn_rate, event_rate, ctmn_period, ctmn_rate,
			event_rate, ctmn_period, register)
	end
end


function fill!(problem::Problem, ::Val{T}, ind::AuxInd) where T
	table = getfield(problem.register.aux, T)
	if haskey(table.dict, ind)
		return
	end
	fill_logic!(problem, Val(T), ind)
end

function fill!(problem::Problem, ::Val{T}, ind::MainInd) where T
	table = getfield(problem.register, T)
	if haskey(table.dict, ind)
		return
	end
	fill_logic!(problem, Val(T), ind)
end


function fill_logic!(problem::Problem, ::Val{:ξ}, ind::AuxInd)
	μ = problem.μ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	ξ = problem.register.aux.ξ

	if r > j
		ξ[ind] = 0.0
	elseif q == 0
		ξ[ind] = -factorial(j) / factorial(r) / (-μ)^(1 + j - r)
	elseif j == 0
		ξ[ind] = 1.0 / μ
	else
		ξ[ind] = ξ(q - 1, r, m, j) - j / μ * ξ(q, r, m, j - 1)
	end
end

function fill_logic!(problem::Problem, ::Val{:ζ}, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	ζ = problem.register.aux.ζ

	if r > j
		ζ[ind] = 0.0
	elseif q == 0
		ζ[ind] = exp(-μ * τ) * factorial(j) / factorial(r) / (-μ)^(1 + j - r)
	elseif j == 0
		ζ[ind] = -exp(-μ * τ) * Polynomial([1 / factorial(k) for k in 0:q])(μ * τ) / μ
	else
		term_1 = (μ * τ)^q / factorial(q) * exp(-μ * τ) / (-μ) * δ(r, j)
		term_2_3 = ζ(q - 1, r, m, j) - j / μ * ζ(q, r, m, j - 1)
		ζ[ind] = term_1 + term_2_3
	end
end

function fill_logic!(problem::Problem, ::Val{:Ξ}, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	Ξ = problem.register.aux.Ξ

	if r > q
		Ξ[ind] = 0.0
	elseif q == 0
		Ξ[ind] = exp(μ * m * τ) * factorial(j) / (-μ)^(1 + j)
	elseif j == 0
		Ξ[ind] = -1 / factorial(r) * μ^(r - 1) * exp(μ * m * τ)
	else
		Ξ[ind] = Ξ(q - 1, r, m, j) - j / μ * Ξ(q, r, m, j - 1)
	end
end

function fill_logic!(problem::Problem, ::Val{:Z}, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	Z = problem.register.aux.Z

	if r > q
		Z[ind] = 0.0
	elseif q == 0
		Z[ind] = -exp(μ * m * τ) * factorial(j) / (-μ)^(1 + j)
	elseif j == 0
		Z[ind] = exp(μ * m * τ) * sum(
			1 / factorial(k) * μ^(k - 1) * binomial(k, r) * τ^(k - r) for k in r:q
		)
	else
		Z[ind] = Z(q - 1, r, m, j) - j / μ * Z(q, r, m, j - 1)
	end
end


function fill_logic!(problem::Problem, ::Val{:η}, ind::AuxInd)
	μ = problem.μ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	η = problem.register.aux.η

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
		η[ind] = μ / (j + 1) * η(q - 1, r, m, j + 1)
	end
end

function fill_logic!(problem::Problem, ::Val{:θ}, ind::AuxInd)
	μ = problem.μ
	τ = problem.τ
	q, r, m, j = ind.q, ind.r, ind.m, ind.j
	θ = problem.register.aux.θ

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
		term_2 = μ / (j + 1) * θ(q - 1, r, m, j + 1)
		θ[ind] = term_1 + term_2
	end
end


function fill_aux!(problem::Problem, n::Int, l::Int)
	"""Fill the auxiliary tables up to the given n and l."""
	aux = problem.register.aux
	qstop, mstop = l, n

	function loop_fill(q::Int, m::Int)
		for j in 0:(n+l-q)
			for r in 0:(q+j)
				ind = (q = q, r = r, m = m, j = j)
				foreach(tbl_sbl -> fill!(problem, Val(tbl_sbl), ind), [:ξ, :ζ, :Ξ, :Z])
			end
		end
		extra = q == 0 ? 1 : 0
		for j in 0:(n+l-q+extra)
			for r in 0:(q+j+1)
				ind = (q = q, r = r, m = m, j = j)
				foreach(tbl_sbl -> fill!(problem, Val(tbl_sbl), ind), [:η, :θ])
			end
		end
	end


	for m in 0:mstop
		for q in 0:qstop
			loop_fill(q, m)
		end
	end



end


function fill_logic!(problem::Problem, ::Val{:a}, ind::MainInd)
	μ = problem.μ
	λ = problem.λ
	τ = problem.τ
	n, l, m, j = ind.n, ind.l, ind.m, ind.j
	a = problem.register.a
	A = problem.register.A
	ξ = problem.register.aux.ξ
	ζ = problem.register.aux.ζ

	if n == 0
		a[ind] = float(δ(m, 0) * δ(j, 0) * δ(l, 0))
	elseif l == 0
		if j == 0
			term_1 = λ * exp(-μ * m * τ) * sum(
						 A(n - 1, 0, m - 1, k) * factorial(k) / (μ^(1 + k)) for k in 0:(n-1)
					 )
			term_2_3 = λ * sum(
				(-a(n - 1, 0, m, k) + a(n - 1, 0, m - 1, k) * exp(-μ * τ)) * factorial(k) / (-μ)^(1 + k) for k in 0:(n-1)
			)
			a[ind] = term_1 + term_2_3
		else
			term_1 = λ * exp(-μ * τ) * a(n - 1, 0, m - 1, j - 1) / j
			term_2_3 = λ * sum(
				(-a(n - 1, 0, m, k) + a(n - 1, 0, m - 1, k) * exp(-μ * τ)) * factorial(k) / factorial(j) / (-μ)^(1 + k - j) for k in j:(n-1)
			)
			a[ind] = term_1 + term_2_3
		end
	else
		if j == 0
			term_1 = λ * exp(-μ * m * τ) * Polynomial(
						 [1 / factorial(q) * sum(A(n - 1, l - q, m - 1, k) * factorial(k) / μ^(k + 1) for k in 0:(n+l-q-1)) for q in 0:l]
					 )(μ * τ)
			term_2_3 = λ * sum(
				sum(a(n - 1, l - q, m, k) * ξ(q, 0, m, k) + a(n - 1, l - q, m - 1, k) * ζ(q, 0, m - 1, k) for k in 0:(n+l-q-1)) for q in 0:l
			)
			a[ind] = term_1 + term_2_3
		else
			term_1 = λ * exp(-μ * τ) * Polynomial(
						 [a(n - 1, l - q, m - 1, j - 1) / j / factorial(q) for q in 0:min(l, n + l - j)]
					 )(μ * τ)
			term_2_3 = λ * sum(
				sum(a(n - 1, l - q, m, k) * ξ(q, j, m, k) + a(n - 1, l - q, m - 1, k) * ζ(q, j, m - 1, k) for k in j:(n+l-q-1)) for q in 0:l
			)
			a[ind] = term_1 + term_2_3
		end
	end
end


function fill_logic!(problem::Problem, ::Val{:A}, ind::MainInd)
	μ = problem.μ
	λ = problem.λ
	τ = problem.τ
	n, l, m, j = ind.n, ind.l, ind.m, ind.j
	a = problem.register.a
	A = problem.register.A
	Ξ = problem.register.aux.Ξ
	Z = problem.register.aux.Z
	η = problem.register.aux.η
	θ = problem.register.aux.θ

	if n == 0
		A[ind] = 0.0
	elseif l == 0
		if j == 0
			term_1 = -λ * sum(
				A(n - 1, 0, m - 1, k) * factorial(k) / μ^(k + 1) for k in 0:(n-1)
			)
			term_2_3 = λ * exp(μ * m * τ) * sum(
						   (a(n - 1, 0, m, k) - a(n - 1, 0, m - 1, k) * exp(-μ * τ)) * factorial(k) / (-μ)^(k + 1) for k in 0:(n-1)
					   )
			A[ind] = term_1 + term_2_3
		else
			term_1 = -λ * sum(
				A(n - 1, 0, m - 1, k) * factorial(k) / factorial(j) / μ^(k + 1 - j) for k in j:(n-1)
			)
			term_2_3 = λ / j * (A(n - 1, 0, m, j - 1) - A(n - 1, 0, m - 1, j - 1))
			A[ind] = term_1 + term_2_3
		end
	else
		term_1 = -λ * Polynomial(
			[1 / factorial(q) * sum(A(n - 1, l - q, m - 1, k) * factorial(k) / factorial(j) / μ^(1 + k - j) for k in j:(n+l-q-1)) for q in 0:l]
		)(μ * τ)
		term_2_3 = λ * sum(
			sum(a(n - 1, l - q, m, k) * Ξ(q, j, m, k) + a(n - 1, l - q, m - 1, k) * Z(q, j, m - 1, k) for k in 0:(n+l-q-1)) for q in j:l
		)
		term_4_5 = λ * sum(
			sum(A(n - 1, l - q, m, k) * η(q, j, m, k) + A(n - 1, l - q, m - 1, k) * θ(q, j, m - 1, k) for k in max(j - q - 1, 0):(n+l-q-1)) for q in 0:l
		)
		A[ind] = term_1 + term_2_3 + term_4_5
	end
end


function fill_main!(problem::Problem, max_n::Int, max_l::Int)
	"""Fill the main tables up to the given max_n and max_l."""
	nstop, lstop = max_n, max_l

	fill_aux!(problem, nstop, lstop)

	function loop_fill(n::Int, l::Int)
		# fill_aux!(problem, n, l)

		for m in 0:n
			for j in 0:(n+l)
				ind = (n = n, l = l, m = m, j = j)
				for tbl_sbl in [:a, :A]
					fill!(problem, Val(tbl_sbl), ind)
				end
			end
		end
	end

	for n in 0:nstop
		for l in 0:lstop
			loop_fill(n, l)
		end
	end

end


function piecewise_monomial(problem::Problem, m::Int, j::Int)
	pow_step(t) = ifelse(t < 0, 0, t^j)
	return t -> pow_step(t - m * problem.τ)
end


function (problem::Problem)(max_n::Int, max_l::Int)
	"""Compute the P probability of the event at the given time."""
	fill_main!(problem, max_n, max_l)

	function inner(observation_time)
		term_1 = exp(-problem.λ * observation_time) * sum(
			problem.register.a(max_n, max_l, m, j) *
			piecewise_monomial(problem, m, j)(observation_time)
			for m in 0:max_n
			for j in 0:(max_n+max_l)
		)
		term_2 =
			exp(-(problem.μ + problem.λ) * observation_time) * sum(
				problem.register.A(max_n, max_l, m, j) *
				piecewise_monomial(problem, m, j)(observation_time)
				for m in 0:max_n
				for j in 0:(max_n+max_l)
			)
		return term_1 + term_2
	end

	return inner
end

end # module
