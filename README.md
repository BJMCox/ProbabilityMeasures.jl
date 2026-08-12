# ProbabilityMeasures

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/ProbabilityMeasures.jl/dev)
[![Test workflow status](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/ProbabilityMeasures.jl)
[![Docs workflow Status](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/rsenne/ProbabilityMeasures.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/rsenne/ProbabilityMeasures.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

A library of normalized probability measures built for probabilistic programming:
type-generic, allocation-free, and clean under automatic differentiation and on the
GPU.

## Why not Distributions.jl

Three things make Distributions.jl an awkward foundation for a PPL. This package
takes a different decision on each:

**Parameters are not promoted.** `Distributions.Normal(μ, σ)` calls `promote`, so an
AD dual entering through `μ` widens `σ` as well, and a `Float32` GPU parameter
silently becomes `Float64` the moment it meets a literal. Here each parameter keeps
its own type:

```julia
julia> typeof(Normal(ForwardDiff.Dual(0.0, 1.0), 1.0))
Normal{ForwardDiff.Dual{Nothing, Float64, 1}, Float64}   # σ untouched

julia> logdensityof(Normal(0, 1), 1.0f0) isa Float32
true                                                     # precision follows x
```

**`logdensityof` is total.** It never throws: correctly-typed `-Inf` outside the
support, `NaN` for invalid parameters. A function that can throw cannot be called
from inside a GPU kernel, and a PPL hands measures invalid parameters constantly
during warmup, line search, and rejected proposals.

**Constructors never validate.** `Normal(0.0, -1.0)` builds without complaint.
`checkparams(d)` is the opt-in check, meant for the boundaries where a human
supplied the numbers, not for the inner loop of a sampler.

## Batching and the GPU

Measures hold scalar, `isbits` parameters, so broadcasting is the batching
mechanism. It fuses into a single kernel, with no wrapper type and no shape
algebra:

```julia
d  = Normal(0.0f0, 1.0f0)
xs = CUDA.randn(Float32, 10^6)
logdensityof.(d, xs)                      # one kernel

mus = CUDA.randn(Float32, 10^6)
logdensityof.(Normal.(mus, 1.0f0), xs)    # still one kernel
```

## Reactant

```julia
using ProbabilityMeasures, Reactant

d  = Normal(0.0, 1.0)
xs = Reactant.to_rarray(randn(1000))
@jit logdensityof.(d, xs)
```

Tracing imposes one constraint the GPU does not: a comparison between traced values
is itself traced, so it cannot drive a branch. Two consequences land in the package
itself, since no extension can retrofit either one.

Measure parameters are bounded by `Number`, not `Real`. Reactant's `TracedRNumber`
is only `<:Number`, so under a `<:Real` bound `Normal{TracedRNumber{Float64},…}`
would not be a constructible type and `μ` or `σ` could not be traced at all. Under
the looser bound nothing checks that a parameter is really real; the package takes
the same line on validation elsewhere.

Branches go through `ProbabilityMeasures.select`, which takes the branch on a `Bool`
and emits a select node on a traced condition. Both arms must be total, since under
tracing the arm not taken is still evaluated. This matters in `logcdf` and
`logccdf`.

`ProbabilityMeasuresReactantExt` holds three methods: the plain float type
underneath a traced one, so `rand` draws its noise unwrapped; `select`; and
`erfcinv`, which Reactant has a `chlo` op for but no `SpecialFunctions` binding to.
Without that last one, `quantile` has nothing to lower to.

Sampling traces too, and stays reparameterized, so Enzyme differentiates a compiled
log-likelihood with respect to `μ` and `σ` and agrees with the analytic gradient to
1e-15. `rand(rng, d, n)` does not trace: Random's sampler machinery fills an
`Array{eltype(d)}`, and a traced number does not convert into a plain `Float64`
slot. Broadcasting is this package's batching mechanism anyway, and the scalar
`rand` it is built from traces fine.

There is also a `ProbabilityMeasuresEnzymeCoreExt`, which marks `checkparams`,
`support`, `insupport`, `noisetype` and `basefloat` inactive so that reverse mode
does not tape predicates and type-level helpers that carry no derivative. It
triggers on EnzymeCore, not Enzyme, so nothing pulls the compiler onto the load
path. There is no Reactant+Enzyme extension: Reactant runs Enzyme on the MLIR and
never consults `EnzymeRules`, so the two never meet.

## The conformance suite

`libs/ProbabilityMeasuresTest` holds a reusable suite that every measure must pass.
Each claim above is checked there:

| Property | Checked with |
| --- | --- |
| Interface conformance | Interfaces.jl |
| Correctness | density integrates to 1 (QuadGK), cdf↔quantile round-trip, Monte Carlo moments |
| Type genericity | `Float32`/`Float64`/`BigFloat` sweep, mixed and exact parameter types |
| Type stability | `@inferred` and JET |
| Allocations | AllocCheck, statically over the whole call graph |
| AD | ForwardDiff, ReverseDiff, Zygote and Mooncake against FiniteDifferences |
| GPU | JLArrays, so scalar-indexing and non-`isbits` capture are caught in CI with no device |
| Reactant | `@jit` over traced data and traced parameters, in its own environment under `test/reactant` |

```julia
using ProbabilityMeasuresTest
test_measure(Normal(0.0, 1.0))
```

It lives in `libs/` and not in the package so that JET, AllocCheck, four AD
backends, JLArrays and QuadGK stay out of the dependency graph. A PPL that depends
on this package stays cheap to load.

## Scope

The exported surface is small: every name is one a PPL is expected to call. Adding
an export later is a non-breaking change and removing one is not, so speculative
names stay out.

Missing so far, and cheap to add when something needs them: `mode`, `skewness`,
`kurtosis`, `mgf`, `cf` (Distributions.jl API surface, not anything inference asks
for); `Matrixvariate`; the `PositiveReals`/`UnitInterval`/`RealInterval` supports,
which will arrive with the first measure that has one.

## Status

Early. `Normal` is implemented and passes the conformance suite. Product and power
measures, transforms, and a Distributions.jl interop extension are next.

## How to Cite

If you use ProbabilityMeasures.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/rsenne/ProbabilityMeasures.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://rsenne.github.io/ProbabilityMeasures.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
