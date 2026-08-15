"""
    Normal(μ, σ)
    Normal()

The normal (Gaussian) measure on ``\\mathbb{R}`` with mean `μ` and standard
deviation `σ`, with density

```math
p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
```

with respect to Lebesgue measure. `Normal()` gives the standard normal in `Float64`.

# Arguments

  - `μ::Number`: the mean.
  - `σ::Number`: the standard deviation.

The `Number` bound permits numeric wrappers used by AD and tracing systems.

# Examples

Construction preserves the types of `μ` and `σ`:

```julia
typeof(Normal(ForwardDiff.Dual(0.0, 1.0), 1.0))
# Normal{ForwardDiff.Dual{Nothing, Float64, 1}, Float64}
```

Integer parameters do not force `Float64`; precision follows the argument:

```julia
logdensityof(Normal(0, 1), 1.0f0) isa Float32     # true
logdensityof(Normal(0, 1), big"1.0") isa BigFloat # true
```

Construction does not validate. Invalid parameters produce a non-finite density;
use [`checkparams`](@ref) to validate explicitly.

```julia
checkparams(Normal(0.0, -1.0))               # false
isnan(logdensityof(Normal(0.0, -1.0), 0.0))  # true
```
"""
struct Normal{M<:Number,S<:Number} <: ContinuousUnivariateMeasure
    μ::M
    σ::S
end

#=
  Julia's generated outer constructor preserves both parameter types. Validation is
  handled by `checkparams`.
=#

# `Float64` here is a default, not a constraint. Write `Normal(0.0f0, 1.0f0)` for Float32.
Normal() = Normal(0.0, 1.0)

Base.eltype(::Type{Normal{M,S}}) where {M,S} = float(promote_type(M, S))

#=
  Use `&` because traced comparisons cannot drive short-circuit evaluation.
=#
checkparams(d::Normal) = isfinite(d.μ) & (d.σ > zero(d.σ))

support(::Normal) = RealLine()

"""
    zval(d::Normal, x)

The standardized value ``(x - \\mu)/\\sigma``.
"""
@inline zval(d::Normal, x::Number) = (x - d.μ) / d.σ

"""
    xval(d::Normal, z)

The inverse of [`zval`](@ref): ``\\mu + \\sigma z``.
"""
@inline xval(d::Normal, z::Number) = muladd(d.σ, z, d.μ)

@inline function DensityInterface.logdensityof(d::Normal, x::Number)
    z = zval(d, x)
    #=
      Convert `σ` to the promoted type of `z` before taking its log. Otherwise an exact
      scale paired with a `BigFloat` location would cap this term at `Float64`.
    =#
    σ = oftype(z, d.σ)
    return -(z^2 + log2π) / 2 - logt(σ)
end

#=
  Draw untracked noise and introduce parameters through arithmetic so pathwise
  gradients need no custom AD rule.
=#
@inline function Base.rand(rng::AbstractRNG, d::Normal)
    return xval(d, randn(rng, noisetype(d)))
end

Statistics.mean(d::Normal) = d.μ
Statistics.var(d::Normal) = d.σ^2

#=
  `abs` keeps `std` consistent with `sqrt(var(d))`, even for an invalid negative scale.
=#
Statistics.std(d::Normal) = abs(d.σ)

Statistics.median(d::Normal) = d.μ

function entropy(d::Normal)
    σ = float(d.σ)
    # `one(σ)` rather than `1`: `log2π + 1` would evaluate the `Irrational` at Float64.
    return (log2π + one(σ)) / 2 + logt(σ)
end

cdf(d::Normal, x::Number) = erfc(-zval(d, x) * invsqrt2) / 2
ccdf(d::Normal, x::Number) = erfc(zval(d, x) * invsqrt2) / 2

#=
  `logerfc` avoids lower-tail underflow; `log1p(-ccdf)` preserves upper-tail precision.
  Use `select` so traced conditions work. Both arms are total over the real line.
=#
function logcdf(d::Normal, x::Number)
    z = zval(d, x)
    return select(
        z < zero(z),
        () -> logerfc(-z * invsqrt2) - logtwo,
        () -> log1p(-erfc(z * invsqrt2) / 2),
    )
end

function logccdf(d::Normal, x::Number)
    z = zval(d, x)
    return select(
        z > zero(z),
        () -> logerfc(z * invsqrt2) - logtwo,
        () -> log1p(-erfc(-z * invsqrt2) / 2),
    )
end

#=
  Negate after multiplication: unary minus would materialize `sqrt2` at `Float64`
  before it can adopt the argument's type. `erfcinvt` keeps out-of-range inputs total.
=#
Statistics.quantile(d::Normal, p::Number) = xval(d, -(sqrt2 * erfcinvt(2 * p)))

function Base.show(io::IO, d::Normal)
    return print(io, "Normal(μ=", d.μ, ", σ=", d.σ, ")")
end
