#=
  The conformance suite. `test_measure` is the single entry point: every measure in
  the package must pass it, and it is where the type-genericity, allocation, AD and
  GPU claims in the README get checked.
=#

"""
    default_ad_backends()

The AD backends exercised by [`test_measure`](@ref) by default.

Enzyme is absent: it is a heavy dependency and its Windows support is uneven, so the
suite would no longer run everywhere. Pass it explicitly via `ad_backends` to include
it.
"""
function default_ad_backends()
    return (
        AutoForwardDiff(), AutoReverseDiff(), AutoZygote(), AutoMooncake(; config=nothing)
    )
end

"Rebuild `d` with parameters taken from the vector `p`."
_reconstruct(d, p) = constructorof(typeof(d))(p...)

#=
  The parameters of `d` as a flat vector, promoted to a common *floating-point*
  type. The `float` matters: a measure may carry integer parameters, and an `Int`
  can neither be perturbed by a finite-difference step nor tracked by AD.
=#
_paramvec(d) = collect(promote(map(float, values(params(d)))...))

"Rebuild `d` with every parameter converted to `T`."
_withtype(d, ::Type{T}) where {T} = _reconstruct(d, map(T, _paramvec(d)))

"""
    test_measure(d; kwargs...)

Run the full conformance suite against the measure `d`.

Each block corresponds to a property this package claims to guarantee. A new measure
is "done" when this passes.

# Keywords

  - `name::AbstractString`: the testset name. Defaults to the type name of `d`.
  - `xs`: evaluation points. Defaults to quantiles spanning the bulk and both tails.
  - `types`: the floating-point types swept for genericity.
  - `ad_backends`: see [`default_ad_backends`](@ref).
  - `reference_logpdf`: an optional `(d, x) -> Real` to check numerics against, for
    example a Distributions.jl equivalent.
  - `nsamples::Int`: Monte Carlo sample count for the moment checks.
  - `check_*::Bool`: force an individual block on or off.

# Which blocks run by default

The blocks that hold for *every* measure (interface conformance, totality, type
genericity, type stability, allocations, AD, GPU broadcast) default to on. A measure
that needs one of them switched off does not conform.

The rest depend on what the measure is, so their defaults are derived from it:

  - `check_normalization` integrates the density with `quadgk`, which is only
    meaningful for a continuous univariate measure. A discrete measure would need a
    sum over its support instead, and nothing here can enumerate one yet.
  - `check_cdf` and `check_moments` cover the *optional* half of `MeasureInterface`,
    so they run only when the measure actually defines those methods.
  - `check_reactant` needs Reactant, a weak dependency here because it brings Enzyme
    and the XLA runtime with it. It runs when the extension has loaded.

Deriving them keeps the defaults from encoding "continuous and univariate" as though
it held for every measure, so the first discrete measure will not have to pass three
`false`s to get a meaningful run.
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
    check_allocations::Bool=true,
    check_normalization::Bool=_can_integrate(d),
    check_cdf::Bool=_has_cdf(d),
    check_moments::Bool=_has_moments(d),
    check_ad::Bool=true,
    check_gpu::Bool=true,
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
  Predicates behind the conditional `check_*` defaults above. They ask what the
  measure *is*, so that adding a discrete or multivariate measure does not require
  editing every `test_measure` call site.
=#

#=
  `test_normalization` and the integral check in `test_cdf` both call `quadgk`
  between `minimum(support(d))` and `maximum(support(d))`. That is a Lebesgue
  integral over an interval: it needs a continuous univariate measure whose support
  has real endpoints.
=#
function _can_integrate(d)
    d isa ContinuousMeasure || return false
    d isa UnivariateMeasure || return false
    s = support(d)
    return hasmethod(minimum, Tuple{typeof(s)}) && hasmethod(maximum, Tuple{typeof(s)})
end

#=
  Whether `f` has a method that genuinely dispatches on `d`.

  `hasmethod` alone is not enough for the moments. `Statistics.mean`, `var` and
  `std` all carry generic *iterator* methods whose argument type is `Any`, so
  `hasmethod(mean, Tuple{typeof(d)})` is true for every measure ever written, even
  one that defines no moments at all. Requiring the resolved method to be narrower
  than `Any` distinguishes "implements mean" from "is a value, and mean accepts
  values", so a measure without moments skips `test_moments` instead of failing
  inside it.
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

function _has_moments(d)
    D = (typeof(d),)
    return _dispatches_on(mean, D) && _dispatches_on(var, D) && _dispatches_on(std, D)
end

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
    for x in (Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64), 0.0)
        @test (logdensityof(d, x); true)
    end
    for x in xs
        @test isfinite(logdensityof(d, x))
    end

    #=
      Invalid parameters produce a non-finite value rather than an error, and
      construction itself never complains.

      Not `isnan`: which non-finite value comes back is not part of the contract.
      `Normal(Inf, 1.0)` is invalid but has a log-density of -Inf, and pinning the
      suite to NaN would push callers toward `isnan` as a validity sentinel. That
      sentinel silently accepts exactly this case.
    =#
    for bad in _invalids(d)
        @test !checkparams(bad)
        @test !isfinite(logdensityof(bad, first(xs)))
    end
end

"Instances of `typeof(d)` with invalid parameters; empty if none are known."
_invalids(d) = ()

# Invariant 1: type genericity.

function test_genericity(d, xs, types)
    for T in types
        dT = _withtype(d, T)
        x = T(first(xs))
        @test logdensityof(dT, x) isa T
        @test rand(Xoshiro(1), dT) isa T
        @test eltype(dT) === T
    end

    #=
      Mixed parameter types must neither error nor widen past the true promotion:
      one Float32 parameter alongside Float64 ones promotes to Float64, and no
      further.

      A *tuple*, not a vector. `[Float32(a), Float64(b)]` is a `Vector{Float64}`:
      the literal promotes and converts the Float32 straight back, so the measure
      comes out homogeneous and the check passes without ever testing anything.
    =#
    p = _paramvec(d)
    if length(p) >= 2
        mixed = _reconstruct(d, (Float32(p[1]), Float64.(p[2:end])...))
        # Assert the measure really is mixed before drawing any conclusion from it.
        @test length(unique(fieldtypes(typeof(mixed)))) > 1
        @test logdensityof(mixed, Float32(first(xs))) isa Float64
    end

    #=
      Exact (integer) parameters must not cap the precision of the result; it has to
      follow the argument.
    =#
    exact = _exactparams(d)
    if exact !== nothing
        x = first(xs)
        @test logdensityof(exact, Float32(x)) isa Float32
        vbig = logdensityof(exact, big(float(x)))
        @test vbig isa BigFloat
        #=
          The same measure with the parameters already widened. If any Irrational
          constant or `log` were evaluated at Float64 along the way, these would
          agree only to ~1e-16 instead of to full BigFloat precision.
        =#
        @test abs(vbig - logdensityof(_withtype(exact, BigFloat), big(float(x)))) < 1e-70
    end
end

"An instance of `typeof(d)` with exact (integer) parameters, or `nothing`."
_exactparams(d) = nothing

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
    s = support(d)
    total, err = quadgk(x -> densityof(d, x), minimum(s), maximum(s); rtol=1e-10)
    @test total ≈ 1 atol = max(1e-8, 10err)
end

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
      log-scale value is still finite.
    =#
    deep = float(quantile(d, 1e-300))
    if isfinite(deep)
        @test isfinite(logcdf(d, deep))
    end

    #=
      The distribution function has to be as type-generic as the density is.
      Checking only `logdensityof` let a `Float64`-collapsing `quantile` through:
      `-sqrt2 * x` parses as `(-sqrt2) * x`, and negating an Irrational materializes
      it at Float64 before it ever sees the argument.
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

    #=
      And as precise. A Float64 intermediate anywhere in the chain caps this at
      ~1e-17 instead of full BigFloat precision.
    =#
    setprecision(BigFloat, 256) do
        dbig = _withtype(d, BigFloat)
        p = big"0.25"
        @test abs(cdf(dbig, quantile(dbig, p)) - p) < 1e-60
    end

    #=
      cdf is the integral of the density for a continuous measure; for a discrete one
      it is a sum. Guard the check on the same predicate that gates
      `test_normalization`.
    =#
    if _can_integrate(d)
        lo = minimum(support(d))
        x = float(quantile(d, 0.3))
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

function test_ad(d, xs, backends)
    x = first(xs)
    p0 = _paramvec(d)
    f = p -> logdensityof(_reconstruct(d, p), x)
    reference = FiniteDifferences.grad(central_fdm(5, 1), f, p0)[1]

    for backend in backends
        @testset "$(nameof(typeof(backend)))" begin
            g = DifferentiationInterface.gradient(f, backend, p0)
            @test g ≈ reference rtol = 1e-5 atol = 1e-8
        end
    end

    #=
      Sampling is written in reparameterized form, so the pathwise derivative must
      exist and be exact. A VI backend in the PPL relies on it.
    =#
    @testset "reparameterized rand" test_reparameterization(d)
end

"Check that `rand` is differentiable with respect to the parameters."
function test_reparameterization(d)
    p0 = _paramvec(d)
    draw = p -> rand(Xoshiro(7), _reconstruct(d, p))
    g = ForwardDiff.gradient(draw, p0)
    reference = FiniteDifferences.grad(central_fdm(5, 1), draw, p0)[1]
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
      JLArray is a CPU-backed GPUArray. It exercises the same broadcast machinery
      and the same scalar-indexing ban as CUDA, so the real GPU failure modes are
      caught on ordinary CI hardware with no device present.
    =#
    d32 = _withtype(d, Float32)
    x32 = Float32.(xs)
    expected = logdensityof.(d32, x32)

    @test isbits(d32)  # a non-isbits measure cannot be captured by a kernel

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
