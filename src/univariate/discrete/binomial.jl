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

`n` is an `Integer` rather than a `Number` because it is structural: it fixes the
support and the loops that run over it, so it can be neither differentiated nor
traced. Precision comes from `p` and from the argument, and `eltype(d)` is
`float(typeof(p))`. The `Number` bound on `p` permits the numeric wrappers used by AD
and tracing systems.

# Cost

The log-density is ``O(1)``. Everything else here is ``O(n)``: `cdf`, `ccdf` and
`quantile` sum the mass function, `entropy` has no closed form, and `rand` adds up `n`
Bernoulli draws. The regularized incomplete beta would give an ``O(1)`` `cdf`, but it
is an iterative continued fraction, so it can be neither traced nor called from a
device kernel. `logcdf` and `logccdf` take the generic `log(cdf(d, x))` fallback,
which underflows further out in the tail than a log-scale sum would.

For large `n` the summed `cdf` reaches one a few atoms short of `n`, where the mass
left is smaller than an ulp of the running total, so `quantile` cannot recover those
atoms from a `cdf` value. It does still return `n` for a probability of one.

Construction does not validate. A `p` outside ``[0, 1]`` gives a non-finite
log-density in the interior of the support but a finite, unnormalized one at `k = 0`
or `k = n`, where the offending factor drops out, so use [`checkparams`](@ref) on
user-supplied probabilities. See [`validateparams`](@ref).

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

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`. There is no zero-argument default: unlike a standard
  normal, no trial count is canonical.
=#

Base.eltype(::Type{Binomial{N,P}}) where {N,P} = float(P)

function checkparams(d::Binomial)
    return (d.n >= zero(d.n)) & isfinite(d.p) & (d.p >= zero(d.p)) & (d.p <= one(d.p))
end

support(d::Binomial) = IntegerRange(0, d.n)

"""
    logbinom(n, k)

``\\log \\binom{n}{k}`` for counts already converted to a float type.

Total: `loggamma` throws for a negative non-integer argument, so the counts are
clamped at zero. A clamped value is only reached outside the support, where the
density is `-Inf` whatever this returns.
"""
@inline function logbinom(n::T, k::T) where {T<:Number}
    nc, kc, mc = max(n, zero(T)), max(k, zero(T)), max(n - k, zero(T))
    return loggamma(nc + one(T)) - loggamma(kc + one(T)) - loggamma(mc + one(T))
end

@inline function DensityInterface.logdensityof(d::Binomial, x::Number)
    T = masstype(d, x)
    n, k, p = convert(T, d.n), convert(T, x), convert(T, d.p)
    #=
      Convert the counts before the log-gammas. They are exact integers, and an `Int`
      argument would compute the coefficient at `Float64` and cap a `BigFloat` density
      there.
    =#
    #=
      A `p` of zero or one is a valid degenerate measure, where a vanishing count has
      to win over an infinite log.
    =#
    a = select(k == zero(T), () -> zero(T), () -> k * logt(p))
    b = select(k == n, () -> zero(T), () -> (n - k) * log1pt(-p))
    return select(insupport(d, x), () -> logbinom(n, k) + a + b, () -> convert(T, -Inf))
end

#=
  Binomial draws have no pathwise derivative, so this is the sum of `n` Bernoulli
  draws rather than anything cleverer. A rejection sampler would be `O(1)` but needs
  to branch on a value.
=#
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
        # An atom of zero mass contributes nothing, where `exp(-Inf) * -Inf` gives `NaN`.
        h -= select(isfinite(logp), () -> exp(logp) * logp, () -> zero(T))
    end
    return h
end

#=
  `cdf` and `ccdf` each sum their own tail rather than complementing the other, so
  neither loses the small one to cancellation. Both clamp: summing every atom can
  overshoot one by an ulp, and a probability above one is worse than a rounded one.
=#
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

#=
  Count the cumulative sums below `q`; a traced value cannot drive an early exit. The
  partial sums come in the same order as in `cdf`, so a `q` that came from `cdf`
  inverts exactly, as long as `cdf` resolved that atom at all.

  For large `n` the sums reach one several atoms below `n`, since the mass out there
  falls below an ulp of the running total. The count then stops early, which is why the
  last atom is handled by the comparison rather than left to the sums.
=#
function Statistics.quantile(d::Binomial, q::Number)
    T = masstype(d, q)
    n = convert(T, d.n)
    total = zero(T)
    i = zero(T)
    for k in 0:(d.n)
        total += exp(logdensityof(d, convert(T, k)))
        i += select(total < q, () -> one(T), () -> zero(T))
    end
    # A `q` at or above one asks for the largest atom; below one, `i` can overshoot it.
    return select(q >= one(T), () -> n, () -> min(i, n))
end

function Base.show(io::IO, d::Binomial)
    return print(io, "Binomial(n=", d.n, ", p=", d.p, ")")
end
