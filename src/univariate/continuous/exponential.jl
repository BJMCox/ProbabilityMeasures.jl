"""
    Exponential(θ)
    Exponential()

The exponential measure on ``[0, \\infty)`` with scale `θ`, with density

```math
p(x) = \\frac{1}{\\theta} \\exp(-x/\\theta)
```

with respect to Lebesgue measure. `θ` is the mean, matching Distributions.jl's
parameterization. `Exponential()` gives the unit-scale exponential in `Float64`.

# Arguments

  - `θ::Number`: the scale.

The `Number` bound permits numeric wrappers used by AD and tracing systems.

Construction does not validate. Invalid parameters produce a non-finite density;
use [`checkparams`](@ref) to validate explicitly.

```julia
checkparams(Exponential(-1.0))               # false
isnan(logdensityof(Exponential(-1.0), 1.0))  # true
```
"""
struct Exponential{R<:Number} <: ContinuousUnivariateMeasure
    θ::R
end

#=
  Julia's generated outer constructor preserves the parameter type. Validation is
  handled by `checkparams`.
=#

# `Float64` here is a default, not a constraint. Write `Exponential(1.0f0)` for Float32.
Exponential() = Exponential(1.0)

Base.eltype(::Type{Exponential{R}}) where {R} = float(R)

checkparams(d::Exponential) = isfinite(d.θ) & (d.θ > zero(d.θ))

support(::Exponential) = NonNegativeReals()

"""
    sval(d::Exponential, x)

The scaled value ``x/\\theta``.
"""
@inline sval(d::Exponential, x::Number) = x / d.θ

@inline function DensityInterface.logdensityof(d::Exponential, x::Number)
    s = sval(d, x)
    θ = oftype(s, d.θ)
    return select(x >= zero(x), () -> -(s + logt(θ)), () -> oftype(s, -Inf))
end

#=
  Draw untracked noise and introduce the scale through multiplication, affine like
  `Normal`'s `xval`, so pathwise gradients need no custom AD rule.
=#
@inline function Base.rand(rng::AbstractRNG, d::Exponential)
    e = -log(rand(rng, noisetype(d)))
    return d.θ * e
end

Statistics.mean(d::Exponential) = d.θ
Statistics.var(d::Exponential) = d.θ^2

#=
  `abs` keeps `std` consistent with `sqrt(var(d))`, even for an invalid negative scale.
=#
Statistics.std(d::Exponential) = abs(d.θ)

function entropy(d::Exponential)
    θ = float(d.θ)
    return one(θ) + logt(θ)
end

function cdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> -expm1(-s), () -> zero(s))
end

function ccdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> exp(-s), () -> one(s))
end

function logcdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> log1mexpt(-s), () -> oftype(s, -Inf))
end

# Exact for every x, unlike `logcdf`: `-s` never needs a stabilizing rewrite.
function logccdf(d::Exponential, x::Number)
    s = sval(d, x)
    return select(x >= zero(x), () -> -s, () -> zero(s))
end

#=
  `log1pt` keeps this total for `p` outside `[0, 1]`, which can arrive from float
  noise in a `cdf` round-trip.
=#
Statistics.quantile(d::Exponential, p::Number) = -d.θ * log1pt(-p)

function Base.show(io::IO, d::Exponential)
    return print(io, "Exponential(θ=", d.θ, ")")
end
