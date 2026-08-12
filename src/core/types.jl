"""
    VariateForm

Trait describing the shape of a single draw from a measure: [`Univariate`](@ref) or
[`Multivariate`](@ref).

There is no `Matrixvariate`. Adding a variate form later is a non-breaking change,
whereas shipping one that nothing implements leaves a downstream PPL supporting a
guess.
"""
abstract type VariateForm end

"Draws are scalars."
struct Univariate <: VariateForm end

"Draws are vectors."
struct Multivariate <: VariateForm end

"""
    ValueSupport

Trait describing whether draws are [`Continuous`](@ref) or [`Discrete`](@ref).
"""
abstract type ValueSupport end

"Draws take values in a continuum, so densities are with respect to Lebesgue measure."
struct Continuous <: ValueSupport end

"Draws take values in a countable set, so densities are with respect to counting measure."
struct Discrete <: ValueSupport end

"""
    AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport}

Supertype for all probability measures.

Every subtype is a *normalized* measure: its density integrates to one against the
measure implied by its [`ValueSupport`](@ref), Lebesgue for [`Continuous`](@ref) and
counting for [`Discrete`](@ref). There is no base-measure recursion and no
unnormalized measure in this package, so `logdensityof` returns the finished value.

# Type parameters

  - `F<:VariateForm`: the shape of a single draw.
  - `S<:ValueSupport`: whether draws are continuous or discrete.

# Required methods

  - `DensityInterface.logdensityof(d, x)`: the normalized log-density.
  - `Base.rand(rng::AbstractRNG, d)`: a single draw.
  - `Base.eltype(::Type{typeof(d)})`: the type of a draw.
  - [`support`](@ref)`(d)`.

[`insupport`](@ref), [`params`](@ref) and the moment functions all have fallbacks.

# Invariants

The conformance suite (`ProbabilityMeasuresTest.test_measure`, in `libs/`) enforces
these, and they must hold:

 1. **Type genericity.** No `Float64` literals in the density. Constants come from
    `IrrationalConstants` or `oftype`. The result type is
    `float(promote_type(<parameter types>..., typeof(x)))`.
 2. **Totality.** `logdensityof` never throws. Outside the support, and for invalid
    parameters, it returns a correctly-typed non-finite value (`-Inf` or `NaN`), so it
    can be called from inside a GPU kernel. Which non-finite value comes back is not
    part of the contract; use [`checkparams`](@ref) rather than `isnan` to detect
    invalid parameters.
 3. **No validation in constructors.** See [`checkparams`](@ref).
 4. **Parameters and arguments are bounded by `Number`, not `Real`.** The measures
    here are real-valued, but several of the wrapper types they must accept are only
    `<:Number`: Reactant's `TracedRNumber` is the current example. A `<:Real` bound
    would make those measures unconstructible, and no extension can widen a bound
    after the fact. Nothing checks that a parameter is really real, which is
    consistent with invariant 3.
 5. **No branching on a value.** A comparison between traced values is itself traced
    and cannot drive `?:`, `&&` or `||`. Use `&`/`|` for predicates and
    `ProbabilityMeasures.select` for a two-way branch, whose arms must both be total.
"""
abstract type AbstractProbabilityMeasure{F<:VariateForm,S<:ValueSupport} end

#=
  Dispatch aliases. `AbstractProbabilityMeasure` is 28 characters, which pushes most
  `<:` clauses past the 92-column margin.

  Only `ContinuousUnivariateMeasure` is exported; it is the supertype for a new
  measure. The other three back the fallbacks in `interface.jl` and remain reachable
  as `ProbabilityMeasures.UnivariateMeasure`.

  There is no `variateform`/`valuesupport` accessor pair either: the parameters are
  already on the type, and `d isa ContinuousMeasure` reads better than
  `valuesupport(d) === Continuous`.
=#
const UnivariateMeasure{S} = AbstractProbabilityMeasure{Univariate,S}
const ContinuousMeasure{F} = AbstractProbabilityMeasure{F,Continuous}
const DiscreteMeasure{F} = AbstractProbabilityMeasure{F,Discrete}
const ContinuousUnivariateMeasure = AbstractProbabilityMeasure{Univariate,Continuous}

#=
  This one line covers batching for univariate measures. Measures hold scalar,
  `isbits` parameters, so `logdensityof.(d, xs)` over a device array captures `d` by
  value and fuses into one kernel, with no wrapper type and no separate batched code
  path.
=#
Base.broadcastable(d::AbstractProbabilityMeasure) = Ref(d)
