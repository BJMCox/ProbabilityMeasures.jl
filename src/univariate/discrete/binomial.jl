"""
    Binomial(n, p)

The binomial measure on ``\\{0, 1, \\ldots, n\\}``, the number of successes in `n`
independent trials of probability `p`, with probability mass function

```math
P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}.
```

# Arguments

  - `n::Integer`: the number of trials.
  - `p::Number`: the success probability of one trial.

`n` stays an integer because it sets the support and loop lengths. The result type
follows `p` and the value being evaluated. Samples use `float(typeof(p))`.

# Cost

The log-density takes constant time. Sampling, entropy, CDFs, and quantiles loop over
`n` trials or outcomes. These loops work with automatic differentiation and on GPUs.

For large `n`, rounding can make the CDF reach one before the last outcome.
`quantile(d, cdf(d, k))` may then fail to recover those final outcomes, though a
probability of one always returns `n`.

The constructor does not check its arguments. An invalid `p` can still give a finite
log-density at `k = 0` or `k = n`, so validate user input with
[`validateparams`](@ref). Use [`checkparams`](@ref) when only a boolean result is
needed.

```julia
checkparams(Binomial(3, 1.5))               # false
isnan(logdensityof(Binomial(3, 1.5), 1.0))  # true
logdensityof(Binomial(3, 1.5), 3.0)         # finite, and wrong
```
"""
struct Binomial{N<:Integer,P<:Number} <: DiscreteUnivariateMeasure
    n::N
    p::P
end

Base.eltype(::Type{Binomial{N,P}}) where {N,P} = float(P)

function checkparams(d::Binomial)
    return (d.n >= zero(d.n)) & isfinite(d.p) & (d.p >= zero(d.p)) & (d.p <= one(d.p))
end

support(d::Binomial) = IntegerRange(0, d.n)

"""
    logbinom(n, k)

Return ``\\log \\binom{n}{k}`` for floating-point counts.

The counts are clamped because `loggamma` rejects negative non-integers. Clamping
only affects values outside the support, where the final density is `-Inf`.
"""
@inline function logbinom(n::T, k::T) where {T<:Number}
    nc, kc, mc = max(n, zero(T)), max(k, zero(T)), max(n - k, zero(T))
    return loggamma(nc + one(T)) - loggamma(kc + one(T)) - loggamma(mc + one(T))
end

@inline function DensityInterface.logdensityof(d::Binomial, x::Number)
    T = masstype(d, x)
    n, k, p = convert(T, d.n), convert(T, x), convert(T, d.p)
    # Convert counts before `loggamma` so an integer `n` does not reduce `BigFloat`
    # precision. Skip zero terms so `p = 0` and `p = 1` work.
    a = select(k == zero(T), () -> zero(T), () -> k * logt(p))
    b = select(k == n, () -> zero(T), () -> (n - k) * log1pt(-p))
    return select(insupport(d, x), () -> logbinom(n, k) + a + b, () -> convert(T, -Inf))
end

# Add `n` Bernoulli samples. This fixed-length loop also works with tracing tools and
# on GPUs.
@inline function Base.rand(rng::AbstractRNG, d::Binomial)
    T = eltype(d)
    s = zero(T)
    for _ in 1:(d.n)
        u = rand(rng, noisetype(d))
        s += select(u < d.p, () -> one(T), () -> zero(T))
    end
    return s
end

Statistics.mean(d::Binomial) = d.n * d.p
Statistics.var(d::Binomial) = d.n * d.p * (one(d.p) - d.p)

# No closed form, so sum over the support.
function entropy(d::Binomial)
    T = eltype(d)
    h = zero(T)
    for k in 0:(d.n)
        logp = logdensityof(d, convert(T, k))
        # An outcome with zero probability contributes zero, not `0 * -Inf = NaN`.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return h
end

# Sum each tail directly so a small tail is not lost by subtracting from one. Clamp
# the result because rounding can make the sum slightly greater than one.
function cdf(d::Binomial, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:(d.n)
        c += select(k <= x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

function ccdf(d::Binomial, x::Number)
    T = masstype(d, x)
    c = zero(T)
    for k in 0:(d.n)
        c += select(k > x, () -> exp(logdensityof(d, convert(T, k))), () -> zero(T))
    end
    return min(c, one(T))
end

# Some tools cannot stop a loop based on `q`, so count every partial sum below it.
# Using the same order as `cdf` makes `quantile(d, cdf(d, k))` return `k` unless the
# CDF has already rounded to one.
function Statistics.quantile(d::Binomial, q::Number)
    T = masstype(d, q)
    n = convert(T, d.n)
    total = zero(T)
    i = zero(T)
    for k in 0:(d.n)
        total += exp(logdensityof(d, convert(T, k)))
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # A probability at or above one always returns the last outcome.
    return select(q >= one(T), () -> n, () -> min(i, n))
end

function Base.show(io::IO, d::Binomial)
    return print(io, "Binomial(n=", d.n, ", p=", d.p, ")")
end
