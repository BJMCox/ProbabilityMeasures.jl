using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Bernoulli(m.p), Int(x))
    for d in (Bernoulli(0.3), Bernoulli(0.5), Bernoulli(0.75f0))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Bernoulli(0.5)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "Bernoulli(p=0.5)"
    @test params(d) === (p=0.5,)
    @test Bernoulli() === Bernoulli(0.5)
end

@testset "no promotion at construction" begin
    @test typeof(Bernoulli(0.5)) === Bernoulli{Float64}
    @test typeof(Bernoulli(0.5f0)) === Bernoulli{Float32}
    @test typeof(Bernoulli(1//2)) === Bernoulli{Rational{Int}}

    # Draws share the type of a density rather than being integers.
    @test eltype(Bernoulli(0.5f0)) === Float32
    @test eltype(Bernoulli(1//2)) === Float64
    @test isbits(Bernoulli(0.5))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Bernoulli(1//2), 1.0f0) isa Float32
    @test logdensityof(Bernoulli(1//2), big"1.0") isa BigFloat

    # Check that no Float64 intermediate caps BigFloat precision.
    exact = logdensityof(Bernoulli(1//2), big"0.0")
    @test abs(exact + log(big"2.0")) < 1e-70
end

@testset "construction never validates" begin
    #=
      A probability outside the unit interval is non-finite at one atom and finite but
      unnormalized at the other, which is the case `validateparams` exists for.
    =#
    below = Bernoulli(-0.5)
    @test !checkparams(below)
    @test isnan(logdensityof(below, 1.0))
    @test isfinite(logdensityof(below, 0.0))

    above = Bernoulli(1.5)
    @test !checkparams(above)
    @test isnan(logdensityof(above, 0.0))
    @test isfinite(logdensityof(above, 1.0))

    @test !checkparams(Bernoulli(NaN))
    @test !checkparams(Bernoulli(Inf))

    @test_throws DomainError validateparams(Bernoulli(1.5))
    @test validateparams(Bernoulli(0.25)) === Bernoulli(0.25)
end

@testset "support" begin
    d = Bernoulli(0.3)
    @test support(d) === IntegerRange(0, 1)
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === 1

    @test insupport(d, 0.0)
    @test insupport(d, 1.0)
    @test insupport(d, 1)
    @test !insupport(d, 2.0)
    @test !insupport(d, -1.0)
    @test !insupport(d, 0.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)
end

@testset "density is total off the support" begin
    d = Bernoulli(0.3)
    for x in (2.0, -1.0, 0.5, Inf, -Inf, NaN, floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    @test isfinite(logdensityof(d, 0.0))
    @test isfinite(logdensityof(d, 1.0))
end

@testset "degenerate probabilities are valid" begin
    @test checkparams(Bernoulli(0.0))
    @test checkparams(Bernoulli(1.0))

    @test logdensityof(Bernoulli(0.0), 0.0) == 0.0
    @test logdensityof(Bernoulli(0.0), 1.0) == -Inf
    @test logdensityof(Bernoulli(1.0), 1.0) == 0.0
    @test logdensityof(Bernoulli(1.0), 0.0) == -Inf

    # A vanishing probability contributes nothing, where `p log p` would give `NaN`.
    @test entropy(Bernoulli(0.0)) == 0.0
    @test entropy(Bernoulli(1.0)) == 0.0
    @test var(Bernoulli(0.0)) == 0.0
    @test var(Bernoulli(1.0)) == 0.0

    @test all(==(0.0), rand(Xoshiro(1), Bernoulli(0.0), 16))
    @test all(==(1.0), rand(Xoshiro(1), Bernoulli(1.0), 16))
end

@testset "reference numerics against Distributions.jl" begin
    for p in (0.0, 0.1, 0.5, 0.9, 1.0)
        d, r = Bernoulli(p), Distributions.Bernoulli(p)
        for x in (0.0, 1.0)
            k = Int(x)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test densityof(d, x) ≈ Distributions.pdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k)
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k)
        end
        for q in (0.0, 0.01, 0.25, 0.5, 0.9, 0.999, 1.0)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test median(d) == Distributions.median(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the atoms" begin
    d = Bernoulli(0.3)
    # The CDF is constant between the atoms.
    @test cdf(d, 0.0) == cdf(d, 0.999) == 0.7
    @test cdf(d, -0.5) == 0.0
    @test cdf(d, 1.0) == cdf(d, 10.0) == 1.0
    @test ccdf(d, 0.0) ≈ 0.3
    @test logcdf(d, -1.0) == -Inf
    @test logcdf(d, 1.0) == 0.0
    @test logccdf(d, -1.0) == 0.0
    @test logccdf(d, 1.0) == -Inf

    # `quantile` inverts `cdf` at each atom.
    @test [quantile(d, cdf(d, x)) for x in (0.0, 1.0)] == [0.0, 1.0]

    # Out-of-range probabilities still return an atom.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "log-density gradient with respect to p" begin
    for p in (0.2, 0.5, 0.9)
        @test ForwardDiff.derivative(q -> logdensityof(Bernoulli(q), 1.0), p) ≈ inv(p)
        @test ForwardDiff.derivative(q -> logdensityof(Bernoulli(q), 0.0), p) ≈ -inv(1 - p)
    end
end

@testset "a draw has no pathwise derivative" begin
    # A Bernoulli draw is piecewise constant in `p`.
    g = ForwardDiff.derivative(q -> rand(Xoshiro(7), Bernoulli(q)), 0.3)
    @test iszero(g)
end

@testset "sampling" begin
    d = Bernoulli(0.3)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Bernoulli(0.3f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 200_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ 0.3 atol = 0.005
end
