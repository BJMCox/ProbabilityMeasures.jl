module ProbabilityMeasuresReverseDiffExt

using ProbabilityMeasures: ProbabilityMeasures
using ReverseDiff: TrackedReal

#=
  Without this, `basefloat` falls through to its `<:Real` fallback, `float(T)`. But
  ReverseDiff overloads `Base.float` on `TrackedReal` to return another tracked value
  rather than stripping tracking, so that fallback resolves to the tracked type
  itself and `randn(rng, ::Type{TrackedReal{...}})` has no method. The noise needs
  the plain type underneath, same reasoning as the ForwardDiff extension.
=#
function ProbabilityMeasures.basefloat(::Type{TrackedReal{V,D,O}}) where {V,D,O}
    return ProbabilityMeasures.basefloat(V)
end

end
