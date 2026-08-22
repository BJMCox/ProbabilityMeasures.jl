using ProbabilityMeasures
using ProbabilityMeasuresTest: test_measure
using Distributions: Distributions
using ForwardDiff: ForwardDiff
using Random: Random, Xoshiro
using Test

#=
  Run the conformance suite across parameter types and signs.
=#
@testset "conformance" begin
    #=
      Distributions.jl is a test-only numerical reference; it shares this
      location-scale parameterization.
    =#
    reference_logpdf(m, x) = Distributions.logpdf(Distributions.Laplace(m.μ, m.b), x)
    for d in (Laplace(0.0, 1.0), Laplace(-2.5, 0.5), Laplace(3.0f0, 2.0f0), Laplace(0, 2))
        test_measure(d; name=string(d), reference_logpdf=reference_logpdf)
    end
end

@testset "no promotion at construction" begin
    dual = ForwardDiff.Dual(0.0, 1.0)
    @test typeof(Laplace(dual, 1.0)) === Laplace{typeof(dual),Float64}
    @test typeof(Laplace(0.0f0, 1)) === Laplace{Float32,Int}
    @test typeof(Laplace(0.0f0, 1.0)) === Laplace{Float32,Float64}

    # A Float32 parameter meeting a Float64 literal must not silently widen.
    @test Laplace(0.0f0, 1.0f0).b isa Float32
end

@testset "precision follows the argument, not the parameters" begin
    # Integer parameters must not pin the result to Float64.
    @test logdensityof(Laplace(0, 2), 1.0f0) isa Float32
    @test logdensityof(Laplace(0, 2), big"1.0") isa BigFloat

    #=
      Check that no Float64 intermediate caps BigFloat precision.
    =#
    exact = logdensityof(Laplace(0, 2), big"1.0")
    full = logdensityof(Laplace(big"0.0", big"2.0"), big"1.0")
    @test abs(exact - full) < 1e-70

    # Rational parameters stay exact, so the `log` term has to float the result.
    @test logdensityof(Laplace(0, 2), 1//2) ≈ -0.25 - log(4.0)
    @test logdensityof(Laplace(0//1, 2//1), 1//2) isa Float64

    #=
      Exact parameters at an exact argument. Rational arithmetic never leaves the exact
      types, so `log(2b)` has to promote before it is taken; a `Float64` intermediate
      would show up as `Rational{BigInt}` disagreeing far above `eps(BigFloat)`.
    =#
    for R in (Rational{Int}, Rational{BigInt})
        F = float(R)
        v = logdensityof(Laplace(R(0), R(2)), R(7) // 5)
        @test v isa F
        @test v ≈ logdensityof(Laplace(F(0), F(2)), F(7) / 5)
    end
end

@testset "construction never validates" begin
    d = Laplace(0.0, -1.0)          # no throw
    @test !checkparams(d)
    @test isnan(logdensityof(d, 0.0))
    @test checkparams(Laplace(0.0, 1.0))
    @test !checkparams(Laplace(Inf, 1.0))
    @test !checkparams(Laplace(0.0, 0.0))
end

@testset "kink at the location" begin
    d = Laplace(1.5, 2.0)
    #=
      The density is continuous at `μ`, so the two one-sided limits meet there, but the
      log-density's derivative jumps from `1/b` to `-1/b`.
    =#
    @test logdensityof(d, 1.5) == -log(4.0)
    @test logdensityof(d, nextfloat(1.5)) ≈ logdensityof(d, 1.5)
    @test logdensityof(d, prevfloat(1.5)) ≈ logdensityof(d, 1.5)

    left = ForwardDiff.derivative(x -> logdensityof(d, x), 1.5 - 1e-6)
    right = ForwardDiff.derivative(x -> logdensityof(d, x), 1.5 + 1e-6)
    @test left ≈ 0.5
    @test right ≈ -0.5
end

@testset "symmetry about the location" begin
    d = Laplace(0.0, 1.5)
    for t in (0.0, 0.3, 1.0, 7.5)
        @test logdensityof(d, t) == logdensityof(d, -t)
        @test cdf(d, -t) ≈ ccdf(d, t)
        @test logcdf(d, -t) ≈ logccdf(d, t)
    end

    # Off centre the reflection is only as exact as `μ + t` and `μ - t` are.
    shifted = Laplace(-2.0, 1.5)
    for t in (0.3, 1.0, 7.5)
        @test logdensityof(shifted, -2.0 + t) ≈ logdensityof(shifted, -2.0 - t)
    end
end

@testset "the location splits the mass in half" begin
    for (μ, b) in ((0.0, 1.0), (-2.5, 0.5), (10.0, 3.0))
        d = Laplace(μ, b)
        @test cdf(d, μ) == 0.5
        @test quantile(d, 0.5) == μ
        @test median(d) == μ
        @test logcdf(d, μ) ≈ -log(2)
    end
end

@testset "closed-form summaries" begin
    for (μ, b) in ((0.0, 1.0), (-2.5, 0.5), (10.0, 3.0))
        d = Laplace(μ, b)
        @test mean(d) == μ
        @test var(d) ≈ 2 * b^2
        @test std(d) ≈ sqrt(2) * b
        @test entropy(d) ≈ 1 + log(2 * b)
    end
end

@testset "log distribution functions beat their logs in the tails" begin
    d = Laplace(0.0, 1.0)
    # Both underflow to exactly zero here; the log-scale values must not.
    @test cdf(d, -800.0) == 0.0
    @test ccdf(d, 800.0) == 0.0
    @test isfinite(logcdf(d, -800.0))
    @test isfinite(logccdf(d, 800.0))
    @test logcdf(d, -800.0) ≈ Distributions.logcdf(Distributions.Laplace(), -800.0)
    @test logccdf(d, 800.0) ≈ Distributions.logccdf(Distributions.Laplace(), 800.0)
end

@testset "reference numerics against Distributions.jl" begin
    ref(μ, b) = Distributions.Laplace(μ, b)
    for (μ, b) in ((0.0, 1.0), (-2.5, 0.5), (10.0, 3.0)), x in (-3.0, -0.4, 0.0, 1.7, 8.0)
        d = Laplace(μ, b)
        @test logdensityof(d, x) ≈ Distributions.logpdf(ref(μ, b), x)
        @test cdf(d, x) ≈ Distributions.cdf(ref(μ, b), x)
        @test ccdf(d, x) ≈ Distributions.ccdf(ref(μ, b), x)
        #=
          `atol` as well: deep in the opposite tail these are tiny negatives, where
          Distributions.jl loses relative accuracy and ours does not.
        =#
        @test logcdf(d, x) ≈ Distributions.logcdf(ref(μ, b), x) atol = 1e-12
        @test logccdf(d, x) ≈ Distributions.logccdf(ref(μ, b), x) atol = 1e-12
    end
    for (μ, b) in ((0.0, 1.0), (-2.5, 0.5)), p in (0.01, 0.25, 0.5, 0.9, 0.999)
        @test quantile(Laplace(μ, b), p) ≈ Distributions.quantile(ref(μ, b), p)
    end
    for (μ, b) in ((0.0, 1.0), (-2.5, 0.5))
        d, r = Laplace(μ, b), ref(μ, b)
        @test mean(d) ≈ Distributions.mean(r)
        @test var(d) ≈ Distributions.var(r)
        @test std(d) ≈ Distributions.std(r)
        @test entropy(d) ≈ Distributions.entropy(r)
    end
end

@testset "sampling" begin
    d = Laplace(1.5, 2.0)
    @test rand(Xoshiro(1), d) isa Float64
    @test rand(Xoshiro(1), Laplace(0.0f0, 1.0f0)) isa Float32
    @test size(rand(Xoshiro(1), d, 3, 4)) == (3, 4)
    @test eltype(rand(Xoshiro(1), d, 5)) === Float64

    v = zeros(4)
    Random.rand!(Xoshiro(1), v, d)
    @test all(isfinite, v)

    #=
      For a reparameterized draw x = μ + b*(e₁ - e₂), d/dμ is one and d/db is the
      underlying noise, recoverable as (x - μ)/b.
    =#
    x = rand(Xoshiro(7), d)
    dμ = ForwardDiff.derivative(m -> rand(Xoshiro(7), Laplace(m, 2.0)), 1.5)
    @test dμ == 1.0
    db = ForwardDiff.derivative(s -> rand(Xoshiro(7), Laplace(1.5, s)), 2.0)
    @test db ≈ (x - 1.5) / 2.0
end
