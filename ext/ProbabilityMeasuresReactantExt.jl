module ProbabilityMeasuresReactantExt

using ProbabilityMeasures: ProbabilityMeasures
using Reactant: Reactant, TracedRNumber
using SpecialFunctions: SpecialFunctions

#=
  Reactant comparisons are traced values and cannot drive Julia branches. These
  methods expose underlying types or replace branches with traced selects.
=#

#=
  `randn(::ReactantRNG, T)` already returns a traced value, so request its plain
  element type rather than nesting `TracedRNumber`.
=#
function ProbabilityMeasures.basefloat(::Type{TracedRNumber{T}}) where {T}
    return ProbabilityMeasures.basefloat(T)
end

#=
  Base's type-level `float` goes through `zero(T)`, which is unavailable for a traced
  type. Remove this method once Reactant provides it upstream.
=#
Base.float(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{float(T)}

#=
  Both arms are evaluated before the traced select, so the `select` contract requires
  total arms.
=#
@inline function ProbabilityMeasures.select(cond::TracedRNumber{Bool}, iftrue, iffalse)
    return ifelse(cond, iftrue(), iffalse())
end

#=
  Reactant lacks bindings for the inverse error functions used by `quantile`, although
  the underlying op exists. Remove these methods once they land upstream.
=#
function SpecialFunctions.erfcinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(one(x) - x)
end

function SpecialFunctions.erfinv(x::TracedRNumber{T}) where {T<:Base.IEEEFloat}
    return Reactant.Ops.erf_inv(x)
end

end
