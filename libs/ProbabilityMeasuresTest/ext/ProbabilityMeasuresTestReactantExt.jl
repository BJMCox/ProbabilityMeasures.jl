module ProbabilityMeasuresTestReactantExt

using ConstructionBase: constructorof
using DensityInterface: logdensityof
using ProbabilityMeasures: AbstractProbabilityMeasure, cdf, ccdf, logcdf, logccdf
using ProbabilityMeasures: checkparams
using ProbabilityMeasuresTest: ProbabilityMeasuresTest
using Random: Random
using Reactant: Reactant, @jit
using Statistics: quantile
using Test: @test, @testset

#=
  The checks jit a broadcast wherever they can, so the result comes back as an `Array`
  and compares directly against the CPU answer. A jitted scalar comes back wrapped in a
  concrete Reactant number, and converting that back to a Julia value is a separate API
  none of these checks are aimed at.
=#

#=
  Parameters go in as separate scalars held in a tuple rather than packed into an array.
  Reactant traverses a tuple and traces each element, so the splat in `_rebuild` stays
  static. Reading parameters back out of a `TracedRArray` would be scalar indexing, which
  Reactant disallows for the same reason a GPU array does.
=#
function _traced_params(d)
    return map(Reactant.ConcreteRNumber, Tuple(ProbabilityMeasuresTest._paramvec(d)))
end

"Rebuild `d` from a tuple of traced parameters."
_rebuild(::D, p) where {D} = constructorof(D)(p...)

#=
  This signature has to be narrower than the `(::Any, ::Any)` stub in `conformance.jl`.
  An identical signature would overwrite the stub instead of adding a method, and method
  overwriting is an error during precompilation.
=#
function ProbabilityMeasuresTest.test_reactant(d::AbstractProbabilityMeasure, xs)
    x = collect(float.(xs))
    rx = Reactant.to_rarray(x)

    @testset "traced data" begin
        #=
          Only the data is traced. The measure stays concrete, so its parameters are
          baked in as constants. A density declared on `x::Real` would already fail
          here: a traced argument is only `<:Number`.
        =#
        @test Array(@jit(logdensityof.(d, rx))) ≈ logdensityof.(d, x)
    end

    @testset "traced parameters" begin
        #=
          Here the parameters are traced too, so the parameter bound has to be `Number`:
          `Normal{TracedRNumber{Float64},…}` must be a constructible type before a
          gradient with respect to μ or σ can be taken under Reactant.
        =#
        f = (pp, xx) -> logdensityof.(_rebuild(d, pp), xx)
        @test Array(@jit(f(_traced_params(d), rx))) ≈ logdensityof.(d, x)
    end

    @testset "distribution functions" begin
        #=
          `logcdf` and `logccdf` branch on the sign of the standardized value, so they
          exercise `select`. `quantile` needs `erfcinv`, which the package's own Reactant
          extension supplies.
        =#
        for f in (cdf, ccdf, logcdf, logccdf)
            @test Array(@jit(f.(d, rx))) ≈ f.(d, x)
        end

        ps = [0.01, 0.25, 0.5, 0.75, 0.99]
        rps = Reactant.to_rarray(ps)
        @test Array(@jit(quantile.(d, rps))) ≈ quantile.(d, ps)
    end

    @testset "totality" begin
        #=
          Invalid parameters must trace to a non-finite number instead of throwing at
          trace time. `test_totality` checks the same invariant on the CPU. A `?:` on a
          traced comparison would fail here.
        =#
        for bad in ProbabilityMeasuresTest._invalids(d)
            f = (pp, xx) -> logdensityof.(_rebuild(bad, pp), xx)
            @test !any(isfinite, Array(@jit(f(_traced_params(bad), rx))))
        end
    end

    @testset "rand" begin
        #=
          The reparameterized draw. `noisetype` decides the type `randn` draws in, so
          `basefloat` has to see through the traced wrapper; asking for the traced type
          would nest one inside another.

          Scalar form only. `rand(rng, d, n)` goes through Random's sampler machinery,
          which fills an `Array{eltype(d)}`, and a traced number does not convert into a
          plain `Float64` slot. Batches go through broadcasting instead; the array form is
          a convenience that tracing does not reach.
        =#
        @test isfinite(Float64(@jit((() -> rand(Random.default_rng(), d))())))

        #= The same draw with the parameters traced, as in pathwise VI. =#
        f = pp -> rand(Random.default_rng(), _rebuild(d, pp))
        @test isfinite(Float64(@jit(f(_traced_params(d)))))
    end

    @testset "checkparams" begin
        #=
          Under tracing the predicate is a traced `Bool`, so it is multiplied through a
          broadcast instead of read as a scalar. What is checked is the value it carries,
          not whether it converts back to a Julia `Bool`.
        =#
        f = (pp, xx) -> checkparams(_rebuild(d, pp)) .* xx
        @test Array(@jit(f(_traced_params(d), rx))) ≈ (checkparams(d) ? x : zero(x))
    end
end

end
