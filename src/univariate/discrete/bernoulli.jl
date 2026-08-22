"""
    Bernoulli(p)
    Bernoulli()

The Bernoulli measure on ``\\{0, 1\\}`` with success probability `p`. Its probability
mass function is

```math
P(X = 1) = p, \\qquad P(X = 0) = 1 - p.
```

`Bernoulli()` creates a fair coin using `Float64`.

# Arguments

  - `p::Number`: the probability of a one.

Samples use the floating-point type of `p`. For example, `Bernoulli(0.5f0)` returns
`Float32` zeros and ones, not integers or booleans.

The constructor does not check `p`. An invalid value can still give a finite
log-density for one outcome, so use [`validateparams`](@ref) for user input. Use
[`checkparams`](@ref) when only a boolean result is needed.

```julia
checkparams(Bernoulli(1.5))               # false
logdensityof(Bernoulli(1.5), 1.0)         # finite, and wrong
isnan(logdensityof(Bernoulli(1.5), 0.0))  # true
```
"""
struct Bernoulli{P<:Number} <: DiscreteUnivariateMeasure
    p::P
end

Bernoulli() = Bernoulli(0.5)

Base.eltype(::Type{Bernoulli{P}}) where {P} = float(P)

checkparams(d::Bernoulli) = isfinite(d.p) & (d.p >= zero(d.p)) & (d.p <= one(d.p))

support(::Bernoulli) = IntegerRange(0, 1)

@inline function DensityInterface.logdensityof(d::Bernoulli, x::Number)
    T = masstype(d, x)
    p = convert(T, d.p)
    return select(
        x == one(x),
        () -> logt(p),
        () -> select(x == zero(x), () -> log1pt(-p), () -> convert(T, -Inf)),
    )
end

# A sample changes in steps as `p` changes, so its derivative is zero almost everywhere.
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
    # A zero probability contributes zero, not `0 * log(0) = NaN`.
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

# The generic log-CDF methods already take the log of `p` or `1 - p` directly.

# Return zero while its probability covers `q`, and one otherwise. Invalid `q` values
# still return one of the two outcomes instead of throwing.
function Statistics.quantile(d::Bernoulli, q::Number)
    T = masstype(d, q)
    p = convert(T, d.p)
    return select(q <= one(T) - p, () -> zero(T), () -> one(T))
end

function Base.show(io::IO, d::Bernoulli)
    return print(io, "Bernoulli(p=", d.p, ")")
end
