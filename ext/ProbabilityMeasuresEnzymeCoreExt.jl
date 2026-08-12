module ProbabilityMeasuresEnzymeCoreExt

using EnzymeCore.EnzymeRules: EnzymeRules
using ProbabilityMeasures: ProbabilityMeasures

#=
  The trigger is EnzymeCore rather than Enzyme. EnzymeCore is the package that carries
  `EnzymeRules`, it is a few hundred lines with no binary dependency, and Enzyme loads
  it, so these rules reach anyone differentiating with Enzyme without pulling the
  compiler onto the load path.

  Enzyme can already differentiate the densities; the arithmetic is plain Julia. The
  rules below only mark the predicates and the type-level helpers as carrying no
  derivative. Without them, an active argument reaching `checkparams` makes reverse mode
  tape a branch on a shadow that is never used, and `noisetype` returns a `Type`, which
  has no shadow to give.
=#

EnzymeRules.inactive(::typeof(ProbabilityMeasures.checkparams), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.support), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.insupport), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.noisetype), args...) = nothing
EnzymeRules.inactive(::typeof(ProbabilityMeasures.basefloat), args...) = nothing

#=
  Supports are singletons (see `Support`), so they hold nothing to differentiate and
  need no shadow allocated for them.
=#
EnzymeRules.inactive_type(::Type{<:ProbabilityMeasures.Support}) = true

end
