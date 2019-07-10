#
# FineShift.jl -
#
# Direct and adjoint operators for sub-sampling separable shift of
# multi-dimensional arrays and for the discrete correlation and
# convolution.
#
module FineShift

export
    convolve!,
    convolve,
    correlate!,
    correlate,
    fineshift,
    fineshift!

using LazyAlgebra
using LinearInterpolators.Kernels

include("impl.jl")

"""

```julia
fineshift!(dst, src, ker, t, d=1, adj=false) -> dst
```

overwrites `dst` with a fine shift of array `src` along the `d`-th dimension.
The shift `t` is given in sampling units and may have a fractional part.  If
`adj` is true, the adjoint of the operator is applied instead.  The fine shift
is performed by interpolating array `src` along its `d`-th dimension with
kernel function `ker`.  Arrays `src` and `dst` must have the same
non-interpolated dimensions (all the dimensions but the `d`-th one).

The result of the operation amounts to doing:

```julia
dst[i,j,k] = sum_jp src[i,jp,k])*ker(j - jp - t)
```

with `ker` the interpolation kernel and for all possible indices.  Indices `i`
and `k` may be multi-dimensional.

For now, only *flat* boundary conditions are implemented.

See also [`fineshift`](@ref), [`correlate!`](@ref),

"""
function fineshift!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                    t::Real,
                    d::Int = 1,
                    opt::Union{Bool,Val} = false) where {T<:AbstractFloat,N,S}
    off, wgt = shiftparams(ker, T(t))
    return correlate!(dst, src, off, wgt, d, opt)
end

"""

```julia
fineshift([len=size(arr,d)], arr, ker, t, d=1, adj=false)
```

yields a fine shift of array `arr` along the `d`-th dimension.  The shift `t`
is given in sampling units and may have a fractional part.  If `adj` is true,
the adjoint of the operator is applied instead.  The fine shift is performed by
interpolating array `arr` along its `d`-th dimension with kernel function
`ker`.  The result is an array whose `d`-th dimension has length `len` and
whose other dimensions have the same length as those of `arr`.

Schematically, assuming `res` is the result of the interpolation and
unidimensional arrays:

```julia
res[i] ≈ arr[i - t]    for i = 1, 2, ..., len
```

See also [`fineshift!`](@ref)

"""
function fineshift(arr::AbstractArray{T,N},
                   ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                   t::Real,
                   args...) where {T<:AbstractFloat,N,S}
    return fineshift!(Array{T,N}(undef, size(arr)), arr, ker, t, args...)
end

function fineshift(len::Int,
                   arr::AbstractArray{T,N},
                   ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                   t::Real,
                   d::Int = 1,
                   args...) where {T<:AbstractFloat,N,S}
    dims = ntuple(i -> (i == d ? len : size(arr, i)), Val(N))
    return fineshift!(Array{T,N}(undef, dims), arr, ker, t, d, args...)
end

"""

```julia
shiftparams(ker, t) -> off, wgt
```

yields the parameters for fine-shifting an array along one of its dimension
by amount `t` and by means of interpolation with kernel `ker`.  The returned
values are `off` an integer offset and `wgt` a `S`-tuple of interpolation
weights with `S` the size of the support of `ker`.

To perform the fine-shifting a source vector `src`, the destination vector
`dst` should be computed as:

```
dst[i] = wgt[1]*src[i-off+1] + ... + wgt[k]*src[i-off+k] +
         ... + wgt[k]*src[i-off+S]
```

that is the sum of `wgt[k]*src[i-off+k]` for `k = 1, ..., S`.

!!! note
    For out of range indices `j = i-off+k`, the value taken by `src[i-off+k]`
    depends on the chosen boundary conditions.

See the note `fast-interpolation.md` for detailed explanations.

"""
shiftparams(ker::Kernel{T,S}, t::Real) where {T<:AbstractFloat,S} =
    shiftparams(ker, T(t))

@generated function shiftparams(ker::Kernel{T,S},
                                t::T) where {T<:AbstractFloat,S}
    W = ntuple(k -> Symbol(:w,k), Val(S))
    c = T(S + 2)/T(2) # 1 + S/2 # variable names for all weights
    quote
        off = floor($c + t)
        v = off - t
        $(ntuple(k -> :($(W[k]) = ker(v - $k)), Val(S))...)
        return (Int(off), ($(W...),))
    end
end

"""

```julia
convolve([len=size(arr,d)], arr, off, wgt, d=1, adj=false, opt=Val(0))
```

yields the discrete convolution of array `arr` by weights `wgt` with offset
`off` along the `d`-th dimension.  The result is an array whose `d`-th
dimension has length `len` and whose other dimensions have the same length
as those of `arr`.

If `adj` is true, the adjoint of the operator is applied instead.

See also [`convolve!`](@ref)

"""
function convolve(arr::AbstractArray{T,N},
                  off::Int,
                  wgt::Tuple{Vararg{T}},
                  args...) where {T<:AbstractFloat,N,S}
    return convolve!(Array{T,N}(undef, size(arr)), arr, off, wgt, args...)
end

function convolve(len::Int,
                  arr::AbstractArray{T,N},
                  off::Int,
                  wgt::Tuple{Vararg{T}},
                  d::Int = 1,
                  args...) where {T<:AbstractFloat,N,S}
    dims = ntuple(i -> (i == d ? len : size(arr, i)), Val(N))
    return convolve!(Array{T,N}(undef, dims), arr, off, wgt, d, args...)
end

"""

```julia
convolve!(dst, src, off, wgt, d=1, adj=false, opt=Val(0)) -> dst
```

overwrites the contents of `dst` with the discrete convolution of `src` by a
kernel whose coefficients are given by `wgt` and with an offset `off` along
the `d`-th dimension.  Flat boundary conditions are assumed.

The call is equivalent to computing:

```
dst[i] = sum_{k=1}^{S} wgt[k]*src[clamp(i-off-k,1,n)]
```

with `S=length(wgt)` and `n=length(x)` for all `i ∈ 1:lenght(y)`.

If argument `adj` is true, the adjoint of the linear operator implemented
by the `convolve!` method is applied.

See also: [`convolve`](@ref), [`correlate!`](@ref).

"""
function convolve!(dst::AbstractArray{T,N},
                   src::AbstractArray{T,N},
                   off::Int,
                   wgt::NTuple{S,T},
                   args...) where {T<:AbstractFloat,N,S}
    return correlate!(dst, src, off, reverse(wgt), args...)
end

"""

```julia
correlate([len=size(arr,d)], arr, off, wgt, d=1, adj=false, opt=Val(0))
```

yields the discrete correlation of array `arr` by weights `wgt` with offset
`off` along the `d`-th dimension.  The result is an array whose `d`-th
dimension has length `len` and whose other dimensions have the same length
as those of `arr`.

If `adj` is true, the adjoint of the operator is applied instead.

See also [`correlate!`](@ref)

"""
function correlate(arr::AbstractArray{T,N},
                   off::Int,
                   wgt::Tuple{Vararg{T}},
                   args...) where {T<:AbstractFloat,N,S}
    return correlate!(Array{T,N}(undef, size(arr)), arr, off, wgt, args...)
end

function correlate(len::Int,
                   arr::AbstractArray{T,N},
                   off::Int,
                   wgt::Tuple{Vararg{T}},
                   d::Int = 1,
                   args...) where {T<:AbstractFloat,N,S}
    dims = ntuple(i -> (i == d ? len : size(arr, i)), Val(N))
    return correlate!(Array{T,N}(undef, dims), arr, off, wgt, d, args...)
end

"""
```julia
correlate!(dst, src, off, wgt, d=1, adj=false, opt=Val(0))
```

overwrite the contents of `dst` with the discrete correlation of `src` by a
kernel whose coefficients are given by `wgt` and with an offset `off` along
the `d`-th dimension.  See [`shiftparams`](@ref) for a description of these
latter arguments.  Flat boundary conditions are assumed.

The call is equivalent to computing:

```
dst[i] = sum_{k=1}^{S} wgt[k]*src[clamp(i-off+k,1,n)]
```

with `S=length(wgt)` and `n=length(x)` for all `i ∈ 1:lenght(y)`.

If argument `adj` is true, the adjoint of the linear operator
implemented by the `correlate!` method is applied.

See also: [`correlate`](@ref), [`fineshift!`](@ref),
[`shiftparams!`](@ref), [`correlate!`](@ref).

"""
function correlate!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    off::Int,
                    wgt::NTuple{S,T},
                    d::Int = 1,
                    adj::Bool = false,
                    opt::Int = 0) where {T<:AbstractFloat,N,S}
    dims, m, n = checkindices(dst, src, d)
    if adj
        fill!(dst, zero(T))
        #@assert m ≥ 1
        Impl._correlate!(Impl.Adjoint(),
                         opt, dst, src, off, wgt, d, dims, n, m)
    else
        #@assert n ≥ 1
        Impl._correlate!(Impl.Direct(),
                         opt, dst, src, off, wgt, d, dims, m, n)
    end
    return dst
end

#------------------------------------------------------------------------------
# UTILITIES

"""

```julia
dimensions(A)
```

yields the list of dimensions of array `A`, like `size(A)` but throws
an error if `A` has non-standard indices.

"""
@inline function dimensions(A::AbstractArray{T,N}) where {T,N}
    inds = axes(A)
    @inbounds for d in 1:N
        first(inds[d]) == 1 || throw_non_standard_indices()
    end
    return size(A)
end

"""
```julia
checkindices(A, B, d) -> dims, m, n
```

checks the indices of arguments `A` and `B` for a separable operation along
their `d`-th dimension and return `dims = size(A)`, `m = size(A,d)` and
`n = size(B,d)`.

"""
@inline function checkindices(A::AbstractArray{<:Any,N},
                              B::AbstractArray{<:Any,N},
                              d::Int) where {N}
    1 ≤ d ≤ N || throw_out_of_range_dimension_index()
    Adims = dimensions(A)
    Bdims = dimensions(B)
    @inbounds begin
        for k in 1:N
            k == d || Adims[k] == Bdims[k] ||
                throw_incompatible_dimensions()
        end
        return (Adims, Adims[d], Bdims[d])
    end
end

#------------------------------------------------------------------------------
# ERRORS

@noinline throw_non_standard_indices() =
    error("array have non standard indices")

@noinline throw_out_of_range_dimension_index() =
    error("out of range dimension index")

@noinline throw_incompatible_dimensions() =
    throw(DimensionMismatch("arrays have incompatible dimensions"))

end # module
