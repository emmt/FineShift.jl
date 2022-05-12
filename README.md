# FineShift Package

| **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![License][License-img]][license-url] | [![Build Status][github-ci-img]][github-ci-url] [![Build Status][appveyor-img]][appveyor-url] | [![Code Coverage][coveralls-img]][coveralls-url] [![Code Coverage][codecov-img]][codecov-url] |

**FineShift** is a [**Julia**][julia-url] package for fast sub-sample
shifting of multi-dimensional arrays.  It can also be used to apply separable
stationary linear filters of small sizes (a.k.a. **discrete correlations**
or **discrete convolutions**).

FineShift implements fine-shifting of Julia arrays by means of separable
interpolation.  After installation (see below), calling `using FineShift`
provides the following methods:

- `fineshift` for fine-shifting along a dimension;
- `fineshift!` is an in-place version of for `fineshift`;
- `convolve` for separable convolution by a small sampled kernel;
- `convolve!` is an in-place version of for `convolve`;
- `correlate` for separable correlation by a small sampled kernel;
- `correlate!` is an in-place version of for `correlate`;


## Installation

The interpolation kernels used by FineShift are provided by the
[`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl)
package which must be installed and some utilities from
[`ArrayTools`](https://github.com/emmt/ArrayTools.jl) are also needed.  .

ArrayTools, InterpolationKernels and FineShift can be installed by Julia's
package manager:

```julia
pkg> add https://github.com/emmt/ArrayTools.jl
pkg> add https://github.com/emmt/InterpolationKernels.jl
pkg> add https://github.com/emmt/FineShift.jl
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/FineShift.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/FineShift.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[github-ci-img]: https://github.com/emmt/FineShift.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/emmt/FineShift.jl/actions/workflows/CI.yml?query=branch%3Amaster

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/FineShift.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/FineShift-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/FineShift.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/FineShift.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/FineShift.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/FineShift.jl?branch=master

[julia-url]: https://pkg.julialang.org/
