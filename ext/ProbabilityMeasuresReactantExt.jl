module ProbabilityMeasuresReactantExt

using ProbabilityMeasures: ProbabilityMeasures
using Reactant: Reactant, TracedRNumber
using SpecialFunctions: SpecialFunctions

#=
  Reactant traces Julia into StableHLO, so it imposes the constraints the package
  already targets for the GPU, plus one: a comparison between traced values is itself a
  traced value and cannot drive a branch. Each method below either supplies the plain
  type underneath a traced one or replaces a branch with a select.

  The densities, the distribution functions, promotion of the `IrrationalConstants`
  constants and the reparameterized `rand` need nothing here. They trace unchanged
  because measures hold their parameters by value and the arithmetic is generic.
=#

#=
  The noise for a reparameterized draw is drawn in the plain element type.
  `randn(::ReactantRNG, T)` already returns a `TracedRNumber{T}`, so asking for the
  traced type here would nest one inside another. Same reasoning as the ForwardDiff
  extension.
=#
function ProbabilityMeasures.basefloat(::Type{TracedRNumber{T}}) where {T}
    return ProbabilityMeasures.basefloat(T)
end

#=
  `eltype` of a measure is `float(promote_type(<parameter types>...))`, evaluated on
  types. Base's generic `float(::Type{<:Number})` goes through `zero(T)`, which a
  traced type has only for values, so the type-level answer has to be given directly.

  This belongs upstream in Reactant, and should be deleted once it lands there.
=#
Base.float(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{float(T)}

#=
  Both arms are evaluated and handed to `stablehlo.select`. The contract on `select`
  requires total arms, so evaluating the arm that is not taken costs work but cannot
  trap.
=#
@inline function ProbabilityMeasures.select(cond::TracedRNumber{Bool}, iftrue, iffalse)
    return ifelse(cond, iftrue(), iffalse())
end

#=
  `ReactantSpecialFunctionsExt` covers `erfc` and `logerfc`, enough for the densities
  and the log-distribution functions, but not the inverses, so `quantile` has nothing to
  lower to. The `chlo` op exists; only the binding to `SpecialFunctions` is missing.
  `erfcinv(y) = erfinv(1 - y)`.

  These two also belong upstream in Reactant, and should be deleted once they land there.
=#
function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(one(x) - x)
end

function SpecialFunctions.erfinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(x)
end

end
