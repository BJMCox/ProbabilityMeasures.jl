module ProbabilityMeasuresReverseDiffExt

using ProbabilityMeasures: ProbabilityMeasures
using ReverseDiff: TrackedReal

#=
  ReverseDiff's `float(TrackedReal)` remains tracked, but sampling noise requires the
  underlying plain type.
=#
function ProbabilityMeasures.basefloat(::Type{TrackedReal{V,D,O}}) where {V,D,O}
    return ProbabilityMeasures.basefloat(V)
end

end
