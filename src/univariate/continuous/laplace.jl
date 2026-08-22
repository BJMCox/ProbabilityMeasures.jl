"""
    Laplace(μ, b)
    Laplace()

The Laplace (double exponential) measure on ``\\mathbb{R}`` with location `μ` and
scale `b`, with density

```math
p(x) = \\frac{1}{2b} \\exp\\!\\left(-\\frac{|x-\\mu|}{b}\\right)
```

with respect to Lebesgue measure. `Laplace()` gives the standard Laplace in `Float64`.

# Arguments

  - `μ::Number`: the location, which is also the mean and the median.
  - `b::Number`: the scale.

The `Number` bound permits numeric wrappers used by AD and tracing systems.

The density is continuous everywhere, with a kink at `x = μ`, where the log-density's
derivative jumps from `1/b` to `-1/b`.

Construction does not validate. Invalid parameters produce a non-finite density;
use [`checkparams`](@ref) to validate explicitly.

```julia
checkparams(Laplace(0.0, -1.0))               # false
isnan(logdensityof(Laplace(0.0, -1.0), 0.0))  # true
```
"""
struct Laplace{M<:Number,B<:Number} <: ContinuousUnivariateMeasure
    μ::M
    b::B
end

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`.
=#

# `Float64` here is a default, not a constraint. Write `Laplace(0.0f0, 1.0f0)` for Float32.
Laplace() = Laplace(0.0, 1.0)

Base.eltype(::Type{Laplace{M,B}}) where {M,B} = float(promote_type(M, B))

checkparams(d::Laplace) = isfinite(d.μ) & isfinite(d.b) & (d.b > zero(d.b))

support(::Laplace) = RealLine()

"""
    zval(d::Laplace, x)

The standardized value ``(x - \\mu)/b``.
"""
@inline zval(d::Laplace, x::Number) = (x - d.μ) / d.b

"""
    xval(d::Laplace, z)

The inverse of [`zval`](@ref): ``\\mu + b z``.
"""
@inline xval(d::Laplace, z::Number) = muladd(d.b, z, d.μ)

@inline function DensityInterface.logdensityof(d::Laplace, x::Number)
    z = zval(d, x)
    #=
      Convert `b` to the promoted type of `z` before taking its log, as `Normal` does
      with `σ`. Otherwise an exact scale paired with a `BigFloat` location would cap
      this term at `Float64`.
    =#
    b = oftype(z, d.b)
    return -abs(z) - logt(2 * b)
end

#=
  A difference of two unit exponentials is exactly Laplace(0, 1). Draw the noise
  untracked and introduce the parameters affinely, so pathwise gradients need no
  custom AD rule.
=#
@inline function Base.rand(rng::AbstractRNG, d::Laplace)
    T = noisetype(d)
    e₁ = -log(rand(rng, T))
    e₂ = -log(rand(rng, T))
    return xval(d, e₁ - e₂)
end

Statistics.mean(d::Laplace) = d.μ
Statistics.median(d::Laplace) = d.μ
Statistics.var(d::Laplace) = 2 * d.b^2

#=
  `abs` keeps `std` consistent with `sqrt(var(d))`, even for an invalid negative scale.
=#
Statistics.std(d::Laplace) = sqrt2 * abs(d.b)

function entropy(d::Laplace)
    b = float(d.b)
    return one(b) + logt(2 * b)
end

#=
  Each half of the density carries one half of the mass. Measuring from the near side
  of `μ` keeps the exponential small, so neither expression cancels.
=#
function cdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> exp(z) / 2, () -> 1 - exp(-z) / 2)
end

function ccdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> exp(-z) / 2, () -> 1 - exp(z) / 2)
end

#=
  Below `μ` the cdf is `exp(z)/2`, so its log is `z - log 2`, exact however deep the
  tail runs. Above `μ` it is one minus a small number, which is what `log1p` is for.
  `log1pt` rather than `log1p` because tracing evaluates both arms, and the untaken
  one runs past `-1`.
=#
function logcdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z < zero(z), () -> z - logtwo, () -> log1pt(-exp(-z) / 2))
end

function logccdf(d::Laplace, x::Number)
    z = zval(d, x)
    return select(z > zero(z), () -> -z - logtwo, () -> log1pt(-exp(z) / 2))
end

#=
  Invert each half against its own endpoint, so the small probability stays the
  argument of the log. `logt` keeps this total for `p` outside `[0, 1]`, which can
  arrive from float noise in a `cdf` round-trip.
=#
function Statistics.quantile(d::Laplace, p::Number)
    half = one(p) / 2
    return select(p < half, () -> xval(d, logt(2 * p)), () -> xval(d, -logt(2 * (1 - p))))
end

function Base.show(io::IO, d::Laplace)
    return print(io, "Laplace(μ=", d.μ, ", b=", d.b, ")")
end
