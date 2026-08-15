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
  Prefer jitted broadcasts so results return as arrays that compare directly with CPU
  results.
=#

#=
  Keep parameters in a tuple so `_rebuild` has a static splat. A `TracedRArray` would
  require forbidden scalar indexing.
=#
function _traced_params(d)
    return map(Reactant.ConcreteRNumber, Tuple(ProbabilityMeasuresTest._paramvec(d)))
end

"Rebuild `d` from a tuple of traced parameters."
_rebuild(::D, p) where {D} = constructorof(D)(p...)

#=
  Keep this signature narrower than the fallback to avoid overwriting it during
  precompilation.
=#
function ProbabilityMeasuresTest.test_reactant(d::AbstractProbabilityMeasure, xs)
    x = collect(float.(xs))
    rx = Reactant.to_rarray(x)

    @testset "traced data" begin
        #=
          Tracing only the data verifies that density arguments accept traced `Number`
          values rather than requiring `Real`.
        =#
        @test Array(@jit(logdensityof.(d, rx))) ≈ logdensityof.(d, x)
    end

    @testset "traced parameters" begin
        #=
          Traced parameters verify that the measure's parameter bounds accept
          `TracedRNumber`.
        =#
        f = (pp, xx) -> logdensityof.(_rebuild(d, pp), xx)
        @test Array(@jit(f(_traced_params(d), rx))) ≈ logdensityof.(d, x)
    end

    @testset "distribution functions" begin
        #=
          Log-CDF functions exercise traced `select`; `quantile` exercises `erfcinv`.
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
          Invalid parameters must trace to non-finite values without throwing.
        =#
        for bad in ProbabilityMeasuresTest._invalids(d)
            f = (pp, xx) -> logdensityof.(_rebuild(bad, pp), xx)
            @test !any(isfinite, Array(@jit(f(_traced_params(bad), rx))))
        end
    end

    @testset "rand" begin
        #=
          Scalar sampling verifies that `basefloat` strips the traced wrapper. Batched
          Random sampling fills a plain array and is outside the tracing path.
        =#
        @test isfinite(Float64(@jit((() -> rand(Random.default_rng(), d))())))

        #= The same draw with the parameters traced, as in pathwise VI. =#
        f = pp -> rand(Random.default_rng(), _rebuild(d, pp))
        @test isfinite(Float64(@jit(f(_traced_params(d)))))
    end

    @testset "checkparams" begin
        #=
          Multiply the traced predicate through a broadcast to test its value without
          converting it to a Julia `Bool`.
        =#
        f = (pp, xx) -> checkparams(_rebuild(d, pp)) .* xx
        @test Array(@jit(f(_traced_params(d), rx))) ≈ (checkparams(d) ? x : zero(x))
    end
end

end
