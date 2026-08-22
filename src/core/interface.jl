#=
  The generic surface every measure inherits. Anything here either has a correct
  fallback or is documented as mandatory in `AbstractProbabilityMeasure`.
=#

DensityInterface.DensityKind(::AbstractProbabilityMeasure) = DensityInterface.HasDensity()

"""
    params(d) -> NamedTuple

The parameters of `d`, keyed by field name.

The default returns the fields as a `NamedTuple`, allowing callers to address
parameters by name.
"""
@inline function StatsAPI.params(d::D) where {D<:AbstractProbabilityMeasure}
    # `D` is a type parameter, so this folds to constants and compiles to a struct load.
    return NamedTuple{fieldnames(D)}(ntuple(i -> getfield(d, i), Val(fieldcount(D))))
end

"""
    checkparams(d) -> Bool

Whether `d`'s parameters are valid, for example a positive scale.

Constructors in this package do not validate. Validation is explicit so construction
can remain usable inside compiled kernels and PPL inner loops.

For invalid parameters, `logdensityof` returns a non-finite value rather than
throwing.

Use this function rather than `isnan(logdensityof(d, x))`: an invalid measure may
produce `-Inf` rather than `NaN`.

# Examples

```julia
d = Normal(0.0, -1.0)     # constructs fine, no error
checkparams(d)            # false
logdensityof(d, 0.0)      # NaN, not a DomainError
```
"""
checkparams(::AbstractProbabilityMeasure) = true

"""
    validateparams(d) -> d

Return `d`, or throw a `DomainError` if [`checkparams`](@ref) rejects its parameters.

For the boundary where user-supplied parameters enter, since constructors do not
validate. It branches on a value and throws, so it can be neither traced nor called from
a device kernel: use it once on the way in, never inside a model.

Reaching for this is worth it where an invalid measure would otherwise go unnoticed.
[`Categorical`](@ref)'s sum-to-one is the case in point: a `p` that does not sum to one
gives a *finite* log-density, too large by `log(sum(p))` for every category, so unlike a
negative scale it does not announce itself. A constant offset also cancels in a
Metropolis-Hastings ratio, which hides it further until `p` starts varying with the
parameters being inferred.

# Examples

```julia
validateparams(Normal(0.0, 1.0))          # returns the measure
validateparams(Categorical([2.0, 2.0]))   # DomainError: does not sum to one
```
"""
function validateparams(d::AbstractProbabilityMeasure)
    checkparams(d) && return d
    throw(DomainError(d, "invalid parameters; see `checkparams`"))
end

"""
    noisetype(d) -> Type{<:AbstractFloat}

The untracked float type used to draw noise for a reparameterized sample from `d`.
"""
@inline function noisetype(d::D) where {D<:AbstractProbabilityMeasure}
    return basefloat(_promoted_paramtype(D))
end

# Promote scalar parameters and array element types.
@inline function _promoted_paramtype(::Type{D}) where {D<:AbstractProbabilityMeasure}
    return promote_type(ntuple(i -> eltype(fieldtype(D, i)), Val(fieldcount(D)))...)
end

"""
    masstype(d, x)

The float type of a probability mass of `d` evaluated at `x`.

Follows the promotion rule in invariant 1 of [`AbstractProbabilityMeasure`](@ref), so a
mass computed in it does not cap the precision of the argument.
"""
@inline function masstype(d::D, x::Number) where {D<:DiscreteMeasure}
    return float(promote_type(_promoted_paramtype(D), typeof(x)))
end

#=
  Array sampling routes through Random's sampler machinery to the scalar
  `Base.rand(rng, d)` implementation.
=#
function Random.rand(
    rng::AbstractRNG, sp::Random.SamplerTrivial{<:AbstractProbabilityMeasure}
)
    return rand(rng, sp[])
end

Base.rand(d::AbstractProbabilityMeasure) = rand(Random.default_rng(), d)

#=
  Extend Statistics for standard summaries; `entropy` is the only package-specific
  summary currently needed by PPL workloads.
=#

"""
    entropy(d)

The differential (or Shannon) entropy of `d`, in nats.
"""
function entropy end

"""
    cdf(d, x)

``P(X \\le x)`` for ``X \\sim d``.
"""
function cdf end

"""
    ccdf(d, x)

``P(X > x) = 1 - ``[`cdf`](@ref)`(d, x)`, computed so that it stays accurate in the
upper tail.
"""
function ccdf end

"""
    logcdf(d, x)

`log(`[`cdf`](@ref)`(d, x))`, computed so that it stays accurate in the lower tail.
"""
function logcdf end

"""
    logccdf(d, x)

`log(`[`ccdf`](@ref)`(d, x))`, computed so that it stays accurate in the upper tail.
"""
function logccdf end

#=
  Generic fallbacks for measures that define the corresponding primitive.
=#
ccdf(d::UnivariateMeasure, x) = one(cdf(d, x)) - cdf(d, x)
logcdf(d::UnivariateMeasure, x) = logt(cdf(d, x))
logccdf(d::UnivariateMeasure, x) = logt(ccdf(d, x))

#=
  Construct one-half in `eltype(d)`; `float(::Rational)` would force `Float64`.
=#
Statistics.median(d::UnivariateMeasure) = quantile(d, one(eltype(d)) / 2)

#=
  A conforming measure has non-negative variance, so plain `sqrt` is total here.
=#
Statistics.std(d::AbstractProbabilityMeasure) = sqrt(var(d))
