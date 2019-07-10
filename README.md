# FineShift Package

| **Documentation**               | **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][doc-dev-img]][doc-dev-url] | [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

**FineShift** is a [**Julia**][julia-url] package for fast sub-sample
shifting of multi-dimensional arrays.  It can also be used to apply separable
stationary linear filters of small sizes (a.k.a. **discrete correlations**
or **discrete convolutions**).

FineShift implements fine-shifting of Julia arrays by means of separable
interpolation.  The interpolation kernels used by FineShift are provided by
the
[`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl)
package which must be installed.


## Installation

InterpolationKernels and FineShift can be installed by Julia's package
manager:

```julia
pkg> add https://github.com/emmt/InterpolationKernels.jl
pkg> add https://github.com/emmt/FineShift.jl
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/FineShift.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/FineShift.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/FineShift.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/FineShift.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/FineShift.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/FineShift-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/FineShift.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/FineShift.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/FineShift.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/FineShift.jl?branch=master

[julia-url]: https://pkg.julialang.org/
