module ProbabilityMeasuresForwardDiffExt

using ForwardDiff: Dual
using ProbabilityMeasures: ProbabilityMeasures

#=
  Draw noise in the underlying float type. Dual parameters enter through the
  reparameterization, so the sample remains differentiable.
=#
function ProbabilityMeasures.basefloat(::Type{<:Dual{T,V,N}}) where {T,V,N}
    return ProbabilityMeasures.basefloat(V)
end

end
