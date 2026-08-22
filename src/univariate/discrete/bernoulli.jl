"""
    Bernoulli(p)
    Bernoulli()

The Bernoulli measure on ``\\{0, 1\\}`` with success probability `p`, with probability
mass function

```math
P(X = 1) = p, \\qquad P(X = 0) = 1 - p.
```

`Bernoulli()` gives the fair coin in `Float64`.

# Arguments

  - `p::Number`: the probability of a one.

The `Number` bound permits numeric wrappers used by AD and tracing systems.

Draws are represented in `eltype(d)`, which is `float(typeof(p))`, so a draw has the
same type as a density rather than being an `Int` or a `Bool`. `Categorical` does the
same.

Construction does not validate. A `p` outside ``[0, 1]`` gives a non-finite
log-density at one atom and a finite, unnormalized one at the other, which is the
situation [`validateparams`](@ref) exists for. Use [`checkparams`](@ref) on
user-supplied probabilities.

```julia
checkparams(Bernoulli(1.5))               # false
logdensityof(Bernoulli(1.5), 1.0)         # finite, and wrong
isnan(logdensityof(Bernoulli(1.5), 0.0))  # true
```
"""
struct Bernoulli{P<:Number} <: DiscreteUnivariateMeasure
    p::P
end

#=
  Julia's generated outer constructor preserves the parameter type. Validation is
  handled by `checkparams`.
=#

# `Float64` here is a default, not a constraint. Write `Bernoulli(0.5f0)` for Float32.
Bernoulli() = Bernoulli(0.5)

Base.eltype(::Type{Bernoulli{P}}) where {P} = float(P)

checkparams(d::Bernoulli) = isfinite(d.p) & (d.p >= zero(d.p)) & (d.p <= one(d.p))

support(::Bernoulli) = IntegerRange(0, 1)

@inline function DensityInterface.logdensityof(d::Bernoulli, x::Number)
    T = masstype(d, x)
    p = convert(T, d.p)
    #=
      One `log` per atom rather than the masked sum `Categorical` needs: with two atoms
      the branch is the whole computation.
    =#
    return select(
        x == one(x),
        () -> logt(p),
        () -> select(x == zero(x), () -> log1pt(-p), () -> convert(T, -Inf)),
    )
end

# Bernoulli draws do not have a pathwise derivative.
@inline function Base.rand(rng::AbstractRNG, d::Bernoulli)
    T = eltype(d)
    u = rand(rng, noisetype(d))
    return select(u < d.p, () -> one(T), () -> zero(T))
end

Statistics.mean(d::Bernoulli) = d.p
Statistics.var(d::Bernoulli) = d.p * (one(d.p) - d.p)

function entropy(d::Bernoulli)
    T = eltype(d)
    p = convert(T, d.p)
    q = one(T) - p
    #=
      A zero probability contributes nothing, where `p log p` would give `NaN`.
    =#
    h = select(p > zero(p), () -> p * logt(p), () -> zero(T))
    return -h - select(q > zero(q), () -> q * log1pt(-p), () -> zero(T))
end

function cdf(d::Bernoulli, x::Number)
    T = masstype(d, x)
    p = convert(T, d.p)
    return select(
        x >= one(x),
        () -> one(T),
        () -> select(x >= zero(x), () -> one(T) - p, () -> zero(T)),
    )
end

function ccdf(d::Bernoulli, x::Number)
    T = masstype(d, x)
    p = convert(T, d.p)
    return select(
        x >= one(x), () -> zero(T), () -> select(x >= zero(x), () -> p, () -> one(T))
    )
end

#=
  No `logcdf` or `logccdf`. Every value the two take is either `p` itself or `1 - p`,
  which floating-point subtraction represents exactly, so the generic
  `logt(cdf(d, x))` fallback is already as accurate as a written-out version.
=#

#=
  The smaller atom whenever its mass reaches `q`. An out-of-range or `NaN` `q` falls
  through the comparison instead of throwing.
=#
function Statistics.quantile(d::Bernoulli, q::Number)
    T = masstype(d, q)
    p = convert(T, d.p)
    return select(q <= one(T) - p, () -> zero(T), () -> one(T))
end

function Base.show(io::IO, d::Bernoulli)
    return print(io, "Bernoulli(p=", d.p, ")")
end
