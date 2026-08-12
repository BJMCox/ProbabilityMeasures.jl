module ProbabilityMeasuresForwardDiffExt

using ForwardDiff: Dual
using ProbabilityMeasures: ProbabilityMeasures

#=
  Without this, `rand(rng, Normal(dual_μ, dual_σ))` would draw the underlying standard
  normal in the dual type, which ForwardDiff does not support. The noise is drawn in the
  plain float type and the duals enter through the reparameterization `μ + σ * z`, so the
  draw stays differentiable in the parameters.
=#
function ProbabilityMeasures.basefloat(::Type{<:Dual{T,V,N}}) where {T,V,N}
    return ProbabilityMeasures.basefloat(V)
end

end
