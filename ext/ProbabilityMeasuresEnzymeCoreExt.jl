module ProbabilityMeasuresEnzymeCoreExt

using EnzymeCore.EnzymeRules: EnzymeRules
using ProbabilityMeasures: ProbabilityMeasures

#=
  Enzyme already differentiates the density arithmetic. Mark predicates and type-level
  helpers inactive because their outputs carry no derivative.
=#

EnzymeRules.inactive(::typeof(ProbabilityMeasures.checkparams), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.support), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.insupport), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.noisetype), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.basefloat), args...) = nothing

# Singleton supports contain no differentiable data.
EnzymeRules.inactive_type(::Type{<:ProbabilityMeasures.Support}) = true

end
