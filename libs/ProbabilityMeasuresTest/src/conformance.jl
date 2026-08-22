#=
  `test_measure` checks the interface, genericity, allocation, AD, and GPU guarantees
  made by the package.
=#

"""
    default_ad_backends()

The AD backends exercised by [`test_measure`](@ref) by default. Enzyme can be passed
explicitly through `ad_backends`.
"""
function default_ad_backends()
    return (
        AutoForwardDiff(), AutoReverseDiff(), AutoZygote(), AutoMooncake(; config=nothing)
    )
end

#=
  Fix the tuple length with `Val(fieldcount(D))`. Splatting `p` directly leaves the
  argument count unknown to inference and breaks Mooncake's static rule builder.
=#
"Rebuild `d` from the flattened parameters in `p`."
function _reconstruct(d, p)
    D = typeof(d)
    offsets, names = _paramoffsets(d), fieldnames(D)
    return constructorof(D)(
        ntuple(
            i -> _unflattenfield(d, names[i], getfield(d, i), p, offsets[i]),
            Val(fieldcount(D)),
        )...,
    )
end

#=
  Flatten parameters into a floating-point vector for numerical differentiation and
  AD.
=#
_paramvec(d) = reduce(vcat, _mapfields(_flattenfield, d))

_flatten(θ::Number) = [float(θ)]
_flatten(θ::AbstractArray) = vec(float.(θ))

_unflatten(::Number, p, offset::Int) = p[offset]
_unflatten(θ::AbstractArray, p, offset::Int) = reshape(p[_range(θ, offset)], size(θ))

_range(θ, offset::Int) = offset:(offset + length(θ) - 1)

#=
  Structural parameters, named per measure. `Binomial`'s `n` fixes loop bounds and the
  support rather than shifting or scaling a density, so it can be neither perturbed nor
  tracked nor converted to another float type. Flattening it to zero entries holds it
  fixed everywhere the others are swept.

  Named rather than recognized by type: an `Integer` parameter is not structural on its
  own, and `_exactparams` deliberately hands ordinary parameters integer values.
=#
"The parameters of `d` that are structural, by name."
_structural(d) = ()

_isstructural(d, name::Symbol) = name in _structural(d)

# Apply `f(d, name, θ)` to each parameter of `d`.
function _mapfields(f, d)
    θs = params(d)
    return map((name, θ) -> f(d, name, θ), keys(θs), values(θs))
end

_flattenfield(d, name::Symbol, θ) = _isstructural(d, name) ? Union{}[] : _flatten(θ)
_lengthfield(d, name::Symbol, θ) = _isstructural(d, name) ? 0 : _paramlength(θ)

function _unflattenfield(d, name::Symbol, θ, p, offset::Int)
    return _isstructural(d, name) ? θ : _unflatten(θ, p, offset)
end

#=
  Preserve structured parameters when flattening and rebuilding them.
=#
_flatten(θ::Diagonal) = _flatten(θ.diag)
_paramlength(θ::Diagonal) = length(θ.diag)
_unflatten(θ::Diagonal, p, offset::Int) = Diagonal(p[_range(θ.diag, offset)])

_flatten(θ::UniformScaling) = [float(θ.λ)]
_paramlength(::UniformScaling) = 1
_unflatten(::UniformScaling, p, offset::Int) = p[offset] * I

"Starting index of each parameter in `_paramvec(d)`."
function _paramoffsets(d)
    lengths = _mapfields(_lengthfield, d)
    #=
      Use a loop because Zygote cannot handle the keyword call required by `sum` here.
    =#
    return ntuple(Val(length(lengths))) do i
        offset = 1
        for j in 1:(i - 1)
            offset += lengths[j]
        end
        return offset
    end
end

_paramlength(::Number) = 1
_paramlength(θ::AbstractArray) = length(θ)

"Rebuild `d` with every parameter converted to `T`."
_withtype(d, ::Type{T}) where {T} = _reconstruct(d, map(T, _paramvec(d)))

# Convert scalar and structured evaluation points to `T`.
_aspoint(x::Number, ::Type{T}) where {T} = T(x)
_aspoint(x::AbstractArray, ::Type{T}) where {T} = T.(x)
_aspoint(x::UniformScaling, ::Type{T}) where {T} = T(x.λ) * I

#=
  Evaluation points as exact rationals. `rationalize` rather than an exact conversion: a
  float converts to a denominator near `2^52`, which overflows `Rational{Int}` as soon as
  a density squares it.
=#
_asexact(x::Number, ::Type{I}) where {I<:Integer} = rationalize(I, x; tol=1//1000)
_asexact(x::AbstractArray, ::Type{I}) where {I<:Integer} = _asexact.(x, I)

# Scalar element type of a draw.
_elscalar(d) = eltype(eltype(d))

# Reduce vector draws to a scalar for gradient tests.
_scalarize(x::Number) = x
_scalarize(x::AbstractArray) = sum(x)

"""
    test_measure(d; kwargs...)

Run the full conformance suite against the measure `d`.

Each block checks a package guarantee.

# Keywords

  - `name::AbstractString`: the testset name. Defaults to the type name of `d`.
  - `xs`: evaluation points. Defaults to quantiles spanning the bulk and both tails.
  - `types`: the floating-point types swept for genericity.
  - `ad_backends`: see [`default_ad_backends`](@ref).
  - `reference_logpdf`: an optional `(d, x) -> Real` to check numerics against, for
    example a Distributions.jl equivalent.
  - `nsamples::Int`: Monte Carlo sample count for the moment checks.
  - `check_*::Bool`: force an individual block on or off.

# Defaults

Interface conformance, totality, type genericity, inference, and AD run for every
measure. Other checks are capability-dependent:

  - normalization uses quadrature for continuous measures and summation for discrete ones;
  - CDF checks run when those optional methods are implemented;
  - the allocation and moment blocks are written around a scalar draw and run for
    univariate measures;
  - GPU checks run for univariate measures;
  - mixed-type checks require more than one scalar parameter;
  - pathwise derivative checks do not run for discrete draws;
  - Reactant checks run when its extension is loaded.

Checks skipped by these defaults belong in the measure's own test file.
"""
function test_measure(
    d;
    name::AbstractString=string(nameof(typeof(d))),
    xs=default_testpoints(d),
    types=(Float32, Float64, BigFloat),
    ad_backends=default_ad_backends(),
    reference_logpdf=nothing,
    nsamples::Int=200_000,
    check_interface::Bool=true,
    check_totality::Bool=true,
    check_genericity::Bool=true,
    check_inference::Bool=true,
    check_allocations::Bool=_can_check_allocations(d),
    check_normalization::Bool=_can_integrate(d) || _can_enumerate(d),
    check_cdf::Bool=_has_cdf(d),
    check_moments::Bool=_can_check_moments(d),
    check_ad::Bool=true,
    check_gpu::Bool=_can_gpu(d),
    check_reactant::Bool=_reactant_loaded(),
)
    @testset "$name" begin
        check_interface && @testset "interface" begin
            #=
              `Interfaces.test` prints a per-component report and returns a Bool;
              without the `@test` the testset records nothing.
            =#
            @test Interfaces.test(MeasureInterface, typeof(d), [d])
        end
        check_totality && @testset "totality" test_totality(d, xs)
        check_genericity && @testset "type genericity" test_genericity(d, xs, types)
        check_inference && @testset "type stability" test_inference(d, xs)
        check_allocations && @testset "allocations" test_allocations(d, xs)
        check_normalization && @testset "normalization" test_normalization(d)
        check_cdf && @testset "distribution function" test_cdf(d, xs)
        check_moments && @testset "moments" test_moments(d, nsamples)
        check_ad && @testset "automatic differentiation" test_ad(d, xs, ad_backends)
        check_gpu && @testset "GPU broadcast" test_gpu(d, xs)
        check_reactant && @testset "Reactant" test_reactant(d, xs)
        if reference_logpdf !== nothing
            @testset "reference numerics" begin
                for x in xs
                    @test logdensityof(d, x) ≈ reference_logpdf(d, x)
                end
            end
        end
    end
end

#=
  Predicates used by the capability-dependent defaults above.
=#

#=
  Quadrature requires a continuous univariate measure whose support provides both
  endpoints.
=#
function _can_integrate(d)
    d isa ContinuousMeasure || return false
    d isa UnivariateMeasure || return false
    return _bounded(support(d))
end

# Discrete normalization requires an enumerable support.
function _can_enumerate(d)
    d isa DiscreteMeasure || return false
    d isa UnivariateMeasure || return false
    return _bounded(support(d))
end

# Ignore Base's generic iterable methods when looking for support bounds.
function _bounded(s)
    return _dispatches_on(minimum, (typeof(s),)) && _dispatches_on(maximum, (typeof(s),))
end

_has_scalar_params(d) = all(T -> T <: Number, fieldtypes(typeof(d)))

# Discrete draws have no pathwise derivative.
_is_reparameterized(d) = !(d isa DiscreteMeasure)

#=
  `hasmethod` also sees Statistics' generic iterator methods on `Any`. Require the
  resolved method to dispatch more narrowly to detect an actual implementation.
=#
function _dispatches_on(f, argtypes::Tuple)
    D = Tuple{argtypes...}
    hasmethod(f, D) || return false
    sig = which(f, D).sig
    params = Base.unwrap_unionall(sig).parameters
    # params[1] is the function's own type; params[2] is the first real argument.
    return length(params) >= 2 && params[2] !== Any
end

#=
  `cdf` and the moments are the optional half of `MeasureInterface`. Probe with
  `eltype(d)` rather than a drawn value: this runs while building default kwargs,
  and must not depend on `rand` having been called.
=#
_has_cdf(d) = _dispatches_on(cdf, (typeof(d), eltype(d)))

#=
  The generic moment checks assume scalar summaries and quantiles.
=#
function _can_check_moments(d)
    d isa UnivariateMeasure || return false
    D = (typeof(d),)
    return _dispatches_on(mean, D) && _dispatches_on(var, D) && _dispatches_on(std, D)
end

#=
  Vector-valued densities and draws may need result storage, so allocation checks are
  defined by their measure-specific tests.
=#
_can_check_allocations(d) = d isa UnivariateMeasure

# The generic GPU test broadcasts over scalar points.
_can_gpu(d) = d isa UnivariateMeasure

"Evaluation points spanning the bulk and both tails of `d`."
function default_testpoints(d)
    ps = (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
    return [float(quantile(d, p)) for p in ps]
end

# Invariant 2: logdensityof is total.

function test_totality(d, xs)
    #=
      A throw here is undefined behaviour inside a GPU kernel, and a PPL will hand
      these values in from a bad proposal or an overshooting line search.
    =#
    for x in _extremepoints(d)
        @test (logdensityof(d, x); true)
    end
    for x in xs
        @test isfinite(logdensityof(d, x))
    end

    #=
      Invalid parameters produce a non-finite value rather than an error. Do not
      require `NaN`: `Normal(Inf, 1.0)` validly produces `-Inf`.
    =#
    for bad in _invalids(d)
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, first(xs)))
    end
end

"Instances of `typeof(d)` with invalid parameters; empty if none are known."
_invalids(d) = ()

"Inputs used to check that `logdensityof` is total."
_extremepoints(d) = (Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64), 0.0)

# Invariant 1: type genericity.

function test_genericity(d, xs, types)
    for T in types
        dT = _withtype(d, T)
        x = _aspoint(first(xs), T)
        @test logdensityof(dT, x) isa T
        @test rand(Xoshiro(1), dT) isa eltype(dT)
        @test _elscalar(dT) === T
    end

    #=
      Build mixed parameters field by field because a vector would promote them first.
    =#
    mixed = _mixedparams(d)
    if mixed !== nothing
        # On element types: two parameters can differ in container but share a scalar type.
        @test length(unique(map(eltype, values(params(mixed))))) > 1
        @test logdensityof(mixed, _aspoint(first(xs), Float32)) isa Float64
    end

    #=
      Exact parameters must not cap the precision of the result; it has to follow the
      argument.
    =#
    exact = _exactparams(d)
    if exact !== nothing
        x = first(xs)
        @test logdensityof(exact, _aspoint(x, Float32)) isa Float32
        vbig = logdensityof(exact, _aspoint(x, BigFloat))
        @test vbig isa BigFloat
        #=
          The same measure with the parameters already widened. If any Irrational
          constant or `log` were evaluated at Float64 along the way, these would
          agree only to ~1e-16 instead of to full BigFloat precision.
        =#
        widened = logdensityof(_withtype(exact, BigFloat), _aspoint(x, BigFloat))
        @test abs(vbig - widened) < 1e-70

        test_exactness(exact, xs)
    end
end

"""
An instance of `typeof(d)` with exact parameters, or `nothing`.

Integers wherever a parameter can take one. A probability cannot: `0` and `1` put a
`-Inf` where the precision checks need a finite value, so a rational serves there
instead.
"""
_exactparams(d) = nothing

#=
  Exact parameters at an exact argument. Rational arithmetic never leaves the exact types,
  so nothing in a density forces a float on the measure's behalf. That is what exposes an
  `Irrational` converted into the argument's own type, which overflows a `Rational{Int}`
  and throws for a `Rational{BigInt}`, and an accumulator declared exact, which the first
  `log` rebinds anyway.

  `Rational{BigInt}` doubles as a precision check: it carries the value exactly all the way
  in, so a `Float64` intermediate shows up as disagreement far above `eps(BigFloat)`.
=#
function test_exactness(exact, xs)
    for I in (Int, BigInt)
        R = Rational{I}
        dR = _withtype(exact, R)
        for x in xs
            xR = _asexact(x, I)
            v = logdensityof(dR, xR)
            @test v isa float(R)
            widened = logdensityof(_withtype(exact, float(R)), _aspoint(xR, float(R)))
            @test isfinite(v) == isfinite(widened)
            isfinite(v) && @test v ≈ widened
        end
    end
end

#=
  Use `Float32` for the first non-structural parameter and `Float64` for the rest.
  Fewer than two of them leaves nothing to mix, as for a measure whose only other
  parameter is structural.
=#
function _mixedparams(d)
    D = typeof(d)
    θs = params(d)
    free = map(name -> !_isstructural(d, name), keys(θs))
    count(free) >= 2 || return nothing
    firstfree = findfirst(free)
    types = ntuple(i -> i == firstfree ? Float32 : Float64, Val(fieldcount(D)))
    # A structural parameter keeps its own type; a float would not rebuild.
    converted = ntuple(Val(fieldcount(D))) do i
        θ = values(θs)[i]
        return free[i] ? _aspoint(θ, types[i]) : θ
    end
    return constructorof(D)(converted...)
end

# Type stability and allocations.

function test_inference(d, xs)
    x = first(xs)
    @test (@inferred logdensityof(d, x)) isa Real
    @test (@inferred rand(Xoshiro(1), d)) isa eltype(d)
    JET.@test_opt target_modules = (ProbabilityMeasures,) logdensityof(d, x)
    JET.@test_call target_modules = (ProbabilityMeasures,) logdensityof(d, x)
end

function test_allocations(d, xs)
    #=
      AllocCheck proves this statically over the whole call graph. `@allocated` would
      only report on the one call it timed, and only if it was warm.
    =#
    @test isempty(check_allocs(logdensityof, (typeof(d), typeof(first(xs)))))
    @test isempty(check_allocs(rand, (Xoshiro, typeof(d))))
end

# Correctness.

function test_normalization(d)
    lo, hi = _quadlimits(d)
    total, err = quadgk(x -> densityof(d, x), lo, hi; rtol=1e-10)
    @test total ≈ 1 atol = max(1e-8, 10err)
end

# Discrete normalization is a sum over the support.
function test_normalization(d::DiscreteMeasure)
    s = support(d)
    @test sum(x -> densityof(d, float(x)), minimum(s):maximum(s)) ≈ 1
end

#=
  Widen integration limits to at least `Float64` so QuadGK can meet the requested
  tolerance.
=#
function _quadlimits(d)
    s = support(d)
    return _widen(minimum(s)), _widen(maximum(s))
end

_widen(x) = convert(promote_type(typeof(float(x)), Float64), x)

function test_cdf(d, xs)
    for x in xs
        c = cdf(d, x)
        @test 0 <= c <= 1
        @test cdf(d, x) + ccdf(d, x) ≈ 1
        @test quantile(d, c) ≈ x rtol = 1e-6
        #=
          `atol` as well as `rtol`: in the upper tail `log(c)` is a tiny negative
          number and a purely relative comparison is meaningless there.
        =#
        @test logcdf(d, x) ≈ log(c) rtol = 1e-8 atol = 1e-12
        @test logccdf(d, x) ≈ log(ccdf(d, x)) rtol = 1e-8 atol = 1e-12
    end

    #=
      `logcdf` exists because `cdf` underflows to zero far out in the tail, where the
      log-scale value is still finite. On a bounded support the deep quantile can round
      onto the lower endpoint, where the cdf really is zero and `-Inf` is the answer, so
      only check strictly inside the support.
    =#
    deep = float(quantile(d, 1e-300))
    if isfinite(deep) && deep > minimum(support(d))
        @test isfinite(logcdf(d, deep))
    end

    #=
      `quantile` must be as total as `logdensityof`: a probability that drifts
      slightly outside `[0, 1]`, for example from float noise in a `cdf` round-trip,
      must not throw.
    =#
    for p in (-0.001, 1.001, -Inf, Inf, NaN)
        @test (quantile(d, p); true)
    end

    #=
      Check distribution functions separately: unary negation of an Irrational can
      otherwise introduce an unnoticed `Float64` intermediate in `quantile`.
    =#
    for T in (Float32, Float64, BigFloat)
        dT = _withtype(d, T)
        xT = T(first(xs))
        @test cdf(dT, xT) isa T
        @test ccdf(dT, xT) isa T
        @test logcdf(dT, xT) isa T
        @test logccdf(dT, xT) isa T
        @test quantile(dT, T(1) / 4) isa T
    end

    # This round trip applies only to continuous CDFs.
    if d isa ContinuousMeasure
        setprecision(BigFloat, 256) do
            dbig = _withtype(d, BigFloat)
            p = big"0.25"
            @test abs(cdf(dbig, quantile(dbig, p)) - p) < 1e-60
        end
    end

    #=
      cdf is the integral of the density for a continuous measure; for a discrete one
      it is a sum. Guard the check on the same predicate that gates
      `test_normalization`.
    =#
    if _can_integrate(d)
        lo = first(_quadlimits(d))
        x = _widen(quantile(d, 0.3))
        integral, _ = quadgk(t -> densityof(d, t), lo, x; rtol=1e-10)
        @test integral ≈ cdf(d, x) rtol = 1e-6
    end
end

function test_moments(d, nsamples)
    rng = Xoshiro(20250801)
    draws = rand(rng, d, nsamples)
    m, v = mean(draws), var(draws)
    # Monte Carlo error on the mean is std/sqrt(n); allow five of them.
    tol = 5 * std(d) / sqrt(nsamples)
    @test m ≈ mean(d) atol = tol
    @test v ≈ var(d) rtol = 20 / sqrt(nsamples)
    @test median(d) ≈ quantile(d, 0.5)
    @test std(d) ≈ sqrt(var(d))
end

# Automatic differentiation.

function test_ad(d, xs, backends; check_reparameterization=_is_reparameterized(d))
    x = first(xs)
    p0 = _paramvec(d)
    #=
      Use at least `Float64` for the finite-difference reference. The AD call still uses
      the measure's parameter type.
    =#
    p0_ref = convert.(promote_type(eltype(p0), Float64), p0)
    f = p -> logdensityof(_reconstruct(d, p), x)
    reference = FiniteDifferences.grad(central_fdm(5, 1), f, p0_ref)[1]

    #=
      Keep the original parameter type so the reference and AD calls draw the same noise.
    =#
    draw = p -> _scalarize(rand(Xoshiro(7), _reconstruct(d, p)))
    draw_reference = if check_reparameterization
        FiniteDifferences.grad(central_fdm(5, 1), draw, p0)[1]
    else
        nothing
    end

    for backend in backends
        @testset "$(nameof(typeof(backend)))" begin
            g = DifferentiationInterface.gradient(f, backend, p0)
            @test g ≈ reference rtol = 1e-5 atol = 1e-8
            if check_reparameterization
                @testset "reparameterized rand" test_reparameterization(
                    draw, p0, draw_reference, backend
                )
            end
        end
    end
end

"Check that `draw` is differentiable with respect to its parameters under `backend`."
function test_reparameterization(draw, p0, reference, backend)
    g = DifferentiationInterface.gradient(draw, backend, p0)
    @test g ≈ reference rtol = 1e-5 atol = 1e-8
    #=
      A zero gradient would mean the draw does not actually depend on the
      parameters, i.e. the reparameterization is broken.
    =#
    @test any(!iszero, g)
end

# GPU.

function test_gpu(d, xs)
    #=
      JLArray exercises GPU broadcast and scalar-indexing rules without requiring a
      physical device.
    =#
    d32 = _withtype(d, Float32)
    x32 = Float32.(xs)
    expected = logdensityof.(d32, x32)

    # Scalar-parameter measures must be capturable by value.
    _has_scalar_params(d) && @test isbits(d32)

    #=
      `allowscalar` only takes a do-block for *permitting* scalar indexing; forbidding
      it means setting the task-local flag directly. The flag is the caller's task
      state, so save and restore it around the broadcast instead of leaving it flipped
      once the suite returns. The idiom mirrors GPUArraysCore's own `@allowscalar`.
    =#
    saved = get(task_local_storage(), :ScalarIndexing, nothing)
    task_local_storage(:ScalarIndexing, GPUArraysCore.ScalarDisallowed)
    try
        got = Array(logdensityof.(d32, JLArray(x32)))
        @test got ≈ expected
        @test eltype(got) === Float32
    finally
        if saved === nothing
            delete!(task_local_storage(), :ScalarIndexing)
        else
            task_local_storage(:ScalarIndexing, saved)
        end
    end
end

# Reactant.

"""
    test_reactant(d, xs)

Check that `d` traces and compiles under Reactant, with the parameters traced as well
as the data.

The method lives in `ext/ProbabilityMeasuresTestReactantExt.jl`. Reactant is a weak
dependency because it brings Enzyme and the XLA runtime with it, a large install for a
suite whose other blocks have no use for them. [`test_measure`](@ref) runs this block
when the extension has loaded and skips it otherwise. Pass `check_reactant=true` to
make its absence an error.
"""
function test_reactant(::Any, ::Any)
    return error("test_reactant needs Reactant. Run `using Reactant` first.")
end

"Whether the Reactant extension of this package has loaded."
function _reactant_loaded()
    return Base.get_extension(@__MODULE__, :ProbabilityMeasuresTestReactantExt) !== nothing
end
