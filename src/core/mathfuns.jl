"""
    select(cond, iftrue, iffalse)

Two-way branch: `iftrue()` when `cond` holds, `iffalse()` otherwise.

A `Bool` condition takes the branch, so the CPU and GPU paths cost what the
equivalent `?:` would. This is a function so that a tracing frontend can extend it:
its comparisons return a traced boolean, which cannot drive a branch, so it adds a
method dispatching on the condition type that evaluates both arms into a select node.
See `ext/ProbabilityMeasuresReactantExt.jl`.

Both arms must therefore be total. Under tracing the arm that is not taken is still
evaluated, so an arm that would throw or trap makes the whole expression unusable.
The thunks keep the `Bool` path from paying for both arms.
"""
@inline select(cond::Bool, iftrue, iffalse) = cond ? iftrue() : iffalse()

"""
    logt(x)

Total `log`: returns `NaN` where `log` would throw a `DomainError`.

`logdensityof` must never throw (invariant 2 of
[`AbstractProbabilityMeasure`](@ref)), because a throw inside a GPU kernel is
undefined behaviour and a PPL will hand these functions invalid parameters during
line search, warmup, and rejected proposals. `log(0)` is already `-Inf` and does not
throw, so only the negative branch needs handling.
"""
@inline function logt(x::Number)
    #=
      `log` is the arm that must stay total under tracing, where both arms evaluate.
      Real `log` throws below zero, but the traced lowering returns NaN there, and
      the select discards it either way.
    =#
    return select(x < zero(x), () -> oftype(float(x), NaN), () -> log(x))
end

"""
    erfcinvt(y)

Total `erfcinv`: returns `NaN` where `erfcinv` would throw a `DomainError`.

[`quantile`](@ref) must stay total (invariant 2 of
[`AbstractProbabilityMeasure`](@ref)): a probability that drifts slightly outside
`[0, 1]`, for example from float noise in a `cdf` round-trip, must not throw.
"""
@inline function erfcinvt(y::Number)
    #=
      `erfcinv`, like `log` in `logt`, is the arm that must stay total under tracing,
      where both arms evaluate. On a concrete `Bool` it is only reached when `y` is
      already in range, so the native domain check never fires.
    =#
    valid = (y >= zero(y)) & (y <= 2 * one(y))
    return select(valid, () -> erfcinv(y), () -> oftype(float(y), NaN))
end

"""
    basefloat(T) -> Type{<:AbstractFloat}

The plain floating-point type underlying `T`, with any AD tracking removed.

Used by [`noisetype`](@ref) to decide the type of the *underlying randomness* in a
reparameterized draw. Sampling `randn` in the tracked type would be wrong, and is
usually unsupported. The tracking enters through the parameters instead, so the draw
is still differentiable.

AD and tracing packages extend this via package extensions; see
`ext/ProbabilityMeasuresForwardDiffExt.jl` and
`ext/ProbabilityMeasuresReactantExt.jl`. The fallbacks below stop at `Real`: a wrapper
type that is only `<:Number` has no correct generic answer here, and a missing method
says so where `float(T)` would quietly return the wrapper.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{T}) where {T<:Real} = float(T)
basefloat(::Type{Bool}) = Float64
basefloat(::Type{<:Irrational}) = Float64
