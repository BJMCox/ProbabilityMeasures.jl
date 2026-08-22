using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

@testset "conformance" begin
    function reference_logpdf(m, x)
        return Distributions.logpdf(Distributions.Binomial(m.n, m.p), Int(x))
    end
    for d in (Binomial(1, 0.5), Binomial(5, 0.3), Binomial(4, 0.6f0), Binomial(20, 0.15))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "traits" begin
    d = Binomial(5, 0.3)
    @test d isa AbstractProbabilityMeasure{Univariate,Discrete}
    @test d isa DiscreteUnivariateMeasure
    @test !(d isa ContinuousUnivariateMeasure)
    @test string(d) == "Binomial(n=5, p=0.3)"
    @test params(d) === (n=5, p=0.3)
end

@testset "the trial count stays fixed" begin
    # `n` stays an integer because it sets the support and loop lengths.
    @test typeof(Binomial(5, 0.3)) === Binomial{Int,Float64}
    @test typeof(Binomial(Int32(5), 0.3f0)) === Binomial{Int32,Float32}
    @test typeof(Binomial(5, 1//2)) === Binomial{Int,Rational{Int}}

    @test eltype(Binomial(5, 0.3f0)) === Float32
    @test eltype(Binomial(5, 1//2)) === Float64
    @test isbits(Binomial(5, 0.3))
end

@testset "precision follows the argument, not the parameters" begin
    @test logdensityof(Binomial(3, 1//2), 1.0f0) isa Float32
    @test logdensityof(Binomial(3, 1//2), big"1.0") isa BigFloat

    # The coefficient must use `BigFloat` even though the counts are integers.
    exact = logdensityof(Binomial(3, 1//2), big"1.0")
    @test abs(exact - (log(big"3.0") - 3 * log(big"2.0"))) < 1e-70
end

@testset "construction never validates" begin
    negative = Binomial(-1, 0.5)
    @test !checkparams(negative)
    @test logdensityof(negative, 0.0) == -Inf

    # An invalid probability can leave an endpoint finite but unnormalized.
    below = Binomial(3, -0.5)
    @test !checkparams(below)
    @test isnan(logdensityof(below, 1.0))
    @test isfinite(logdensityof(below, 0.0))

    above = Binomial(3, 1.5)
    @test !checkparams(above)
    @test isnan(logdensityof(above, 1.0))
    @test isfinite(logdensityof(above, 3.0))

    @test !checkparams(Binomial(3, NaN))
    @test !checkparams(Binomial(3, Inf))

    @test_throws DomainError validateparams(Binomial(3, 1.5))
    @test validateparams(Binomial(3, 0.25)) === Binomial(3, 0.25)
end

@testset "support" begin
    d = Binomial(5, 0.3)
    @test support(d) === IntegerRange(0, 5)
    @test minimum(support(d)) === 0
    @test maximum(support(d)) === 5

    @test insupport(d, 0.0)
    @test insupport(d, 5.0)
    @test insupport(d, 3)
    @test !insupport(d, -1.0)
    @test !insupport(d, 6.0)
    @test !insupport(d, 2.5)
    @test !insupport(d, NaN)
    @test !insupport(d, Inf)

    @test support(Binomial(-1, 0.5)) === IntegerRange(0, -1)
end

@testset "density is total off the support" begin
    d = Binomial(5, 0.3)
    # Counts are clamped before `loggamma` so values outside the support do not throw.
    for x in (-1.0, -1.5, 6.0, 2.5, Inf, -Inf, NaN, floatmax(Float64), -floatmax(Float64))
        @test logdensityof(d, x) == -Inf
    end
    for k in 0:5
        @test isfinite(logdensityof(d, float(k)))
    end
end

@testset "n = 1 is Bernoulli" begin
    for p in (0.0, 0.25, 0.5, 1.0)
        b, d = Bernoulli(p), Binomial(1, p)
        for x in (0.0, 1.0, 2.0, -1.0, 0.5)
            @test logdensityof(d, x) == logdensityof(b, x)
        end
        @test cdf(d, 0.0) ≈ cdf(b, 0.0)
        @test mean(d) ≈ mean(b)
        @test var(d) ≈ var(b)
        @test entropy(d) ≈ entropy(b)
    end
end

@testset "degenerate parameters" begin
    # No trials puts all the mass on zero successes.
    d = Binomial(0, 0.3)
    @test checkparams(d)
    @test support(d) === IntegerRange(0, 0)
    @test logdensityof(d, 0.0) == 0.0
    @test logdensityof(d, 1.0) == -Inf
    @test mean(d) == 0.0
    @test var(d) == 0.0
    @test entropy(d) == 0.0
    @test all(==(0.0), rand(Xoshiro(1), d, 8))

    # Zero terms must win over the infinite log at `p = 0` or `p = 1`.
    @test logdensityof(Binomial(3, 0.0), 0.0) == 0.0
    @test logdensityof(Binomial(3, 0.0), 1.0) == -Inf
    @test logdensityof(Binomial(3, 1.0), 3.0) == 0.0
    @test logdensityof(Binomial(3, 1.0), 2.0) == -Inf
    @test entropy(Binomial(3, 0.0)) == 0.0
    @test all(==(3.0), rand(Xoshiro(1), Binomial(3, 1.0), 8))
end

@testset "reference numerics against Distributions.jl" begin
    for (n, p) in ((1, 0.5), (5, 0.3), (10, 0.5), (20, 0.9), (50, 0.3))
        d, r = Binomial(n, p), Distributions.Binomial(n, p)
        for k in 0:n
            x = float(k)
            @test logdensityof(d, x) ≈ Distributions.logpdf(r, k)
            @test densityof(d, x) ≈ Distributions.pdf(r, k)
            @test cdf(d, x) ≈ Distributions.cdf(r, k)
            @test ccdf(d, x) ≈ Distributions.ccdf(r, k)
            # Relative error is not useful when these log values are near zero.
            @test logcdf(d, x) ≈ Distributions.logcdf(r, k) atol = 1e-12
            @test logccdf(d, x) ≈ Distributions.logccdf(r, k) atol = 1e-12
        end
        for q in (0.0, 0.01, 0.25, 0.5, 0.9, 0.999, 1.0)
            @test quantile(d, q) == Distributions.quantile(r, q)
        end
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "distribution functions step at the atoms" begin
    d = Binomial(5, 0.3)
    # The CDF is constant between the atoms.
    @test cdf(d, 2.0) == cdf(d, 2.999)
    @test cdf(d, -0.5) == 0.0
    @test cdf(d, 5.0) ≈ 1.0
    @test cdf(d, 10.0) ≈ 1.0
    @test logcdf(d, -1.0) == -Inf
    @test logccdf(d, 5.0) == -Inf

    # Each tail is summed on its own, so the two agree without cancelling.
    for k in 0:5
        @test cdf(d, float(k)) + ccdf(d, float(k)) ≈ 1.0
    end

    @test [quantile(d, cdf(d, float(k))) for k in 0:5] == float.(0:5)

    # Out-of-range probabilities still return an atom.
    for q in (-0.001, 1.001, -Inf, Inf, NaN)
        @test insupport(d, quantile(d, q))
    end
end

@testset "log-density gradient with respect to p" begin
    n = 5
    for p in (0.2, 0.5, 0.9), k in 0:n
        g = ForwardDiff.derivative(q -> logdensityof(Binomial(n, q), float(k)), p)
        @test g ≈ k / p - (n - k) / (1 - p)
    end
end

@testset "sample derivative is zero" begin
    # A sample changes in steps as `p` changes.
    g = ForwardDiff.derivative(q -> rand(Xoshiro(7), Binomial(5, q)), 0.3)
    @test iszero(g)
end

@testset "sampling" begin
    d = Binomial(5, 0.3)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Binomial(5, 0.3f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(x -> insupport(d, x), v)

    draws = rand(Xoshiro(20250801), d, 200_000)
    @test all(x -> insupport(d, x), draws)
    @test mean(draws) ≈ mean(d) atol = 0.01
    @test var(draws) ≈ var(d) atol = 0.02
end
