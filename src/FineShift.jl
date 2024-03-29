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

using ArrayTools
using InterpolationKernels

include("impl.jl")

"""
    fineshift!(dst, src, ker, t, d=1, adj=false) -> dst

overwrites `dst` with array `src` shifted along its `d`-th dimension.  The
shift `t` is given in sampling units and may have a fractional part.  The shift
is performed by interpolating array `src` along its `d`-th dimension with
kernel function `ker`.  Arrays `src` and `dst` must have the same
non-interpolated dimensions (all dimensions but the `d`-th one).

The result of the operation amounts to doing:

    dst[i,j,k] = sum_jp src[i,jp,k])*ker(j - jp - t)

with `ker` the interpolation kernel and for all possible indices.  Indices `i`
and `k` may be multi-dimensional (including 0-dimensional).

For now, only *flat* boundary conditions are implemented.

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

See also [`fineshift`](@ref), [`correlate!`](@ref),

"""
function fineshift!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    ker::Kernel{T,S},
                    t::Real,
                    d::Int = 1,
                    opt::Union{Bool,Val} = false) where {T<:AbstractFloat,N,S}
    off, wgt = shiftparams(ker, T(t))
    return correlate!(dst, src, off, wgt, d, opt)
end

"""
    fineshift([len=size(arr,d)], arr, ker, t, d=1, adj=false)

yields array `arr` shifted along its `d`-th dimension.  The shift `t` is given
in sampling units and may have a fractional part.  The shift is performed by
interpolating array `arr` along its `d`-th dimension with kernel function
`ker`.  The result is an array whose `d`-th dimension has length `len` and
whose other dimensions have the same length as those of `arr`.

Schematically, assuming `res` is the result of the interpolation and
unidimensional arrays:

    res[i] ≈ arr[i - t]    for i = 1, 2, ..., len

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

For now, only *flat* boundary conditions are implemented.

See also [`fineshift!`](@ref)

"""
function fineshift(arr::AbstractArray{T,N},
                   ker::Kernel{T,S},
                   t::Real,
                   args...) where {T<:AbstractFloat,N,S}
    return fineshift!(Array{T,N}(undef, size(arr)), arr, ker, t, args...)
end

function fineshift(len::Int,
                   arr::AbstractArray{T,N},
                   ker::Kernel{T,S},
                   t::Real,
                   d::Int = 1,
                   args...) where {T<:AbstractFloat,N,S}
    dims = ntuple(i -> (i == d ? len : size(arr, i)), Val(N))
    return fineshift!(Array{T,N}(undef, dims), arr, ker, t, d, args...)
end

"""
    shiftparams(ker, t) -> off, wgt

yields the parameters for fine-shifting an array along one of its dimension by
amount `t` and by means of interpolation with kernel `ker`.  The returned
values are `off`, an integer offset, and `wgt`, a `S`-tuple of interpolation
weights with `S` the size of the support of `ker`.

To perform the fine-shifting a source vector `src`, the destination vector
`dst` should be computed as:

    dst[i] = wgt[1]*src[i-off+1] + ... + wgt[k]*src[i-off+k] +
             ... + wgt[S]*src[i-off+S]

that is the sum of `wgt[k]*src[i-off+k]` for `k = 1, ..., S` with `S` the
number of interpolation weights.

!!! note
    For out of range indices `k = i-off+k`, the value taken by `src[i-off+k]`
    depends on the chosen boundary conditions.

See the note `fast-interpolation.md` for detailed explanations.

"""
function shiftparams(ker::Kernel{T}, t::Real) where {T}
    off, wgt = InterpolationKernels.compute_offset_and_weights(ker, -T(t))
    return safe_int(-off), wgt
end

# Convert `x` to an integer of type `Int` taking care of overflows but not
# of "inexact errors".
safe_int(x::Int) = x
safe_int(x::Real) = (x ≤ typemin(Int) ? typemin(Int) :
                     x ≥ typemax(Int) ? typemax(Int) : Int(x))

"""
    convolve([len=size(arr,d)], arr, off, wgt, d=1, adj=false, opt=Val(0))

yields the discrete convolution of array `arr` by weights `wgt` with offset
`off` along the `d`-th dimension.  The result is an array whose `d`-th
dimension has length `len` and whose other dimensions have the same length as
those of `arr`.

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

For now, only *flat* boundary conditions are implemented.

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
    convolve!(dst, src, off, wgt, d=1, adj=false, opt=Val(0)) -> dst

overwrites the contents of `dst` with the discrete convolution of `src` by a
kernel whose coefficients are given by `wgt` and with an offset `off` along the
`d`-th dimension.  Flat boundary conditions are assumed.

The call is equivalent to computing:

    dst[i] = sum_{k ∈ 1:S} wgt[k]*src[clamp(i-off-k,1,n)]

with `S = length(wgt)` and `n = length(x)` for all `i ∈ 1:length(dst)`.

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

See also: [`convolve`](@ref), [`correlate!`](@ref).

"""
function convolve!(dst::AbstractArray{T,N},
                   src::AbstractArray{T,N},
                   off::Int,
                   wgt::NTuple{S,T},
                   args...) where {T<:AbstractFloat,N,S}
    # We really want to compute a discrete correlation because addressing array
    # entries in increasing order is the key for vectorization.
    return correlate!(dst, src, off, reverse(wgt), args...)
end

"""
    correlate([len=size(arr,d)], arr, off, wgt, d=1, adj=false, opt=Val(0))

yields the discrete correlation of array `arr` by weights `wgt` with offset
`off` along the `d`-th dimension.  The result is an array whose `d`-th
dimension has length `len` and whose other dimensions have the same length as
those of `arr`.

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

For now, only *flat* boundary conditions are implemented.

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
    correlate!(dst, src, off, wgt, d=1, adj=false, opt=Val(0))

overwrite the contents of `dst` with the discrete correlation of `src` by a
kernel whose coefficients are given by `wgt` and with an offset `off` along the
`d`-th dimension.  See [`shiftparams`](@ref) for a description of these latter
arguments.  Flat boundary conditions are assumed.

The call is equivalent to computing:

    dst[i] = sum_{k ∈ 1:S} wgt[k]*src[clamp(i-off+k,1,n)]

with `S = length(wgt)` and `n = length(x)` for all `i ∈ 1:length(dst)`.

If optional argument `adj` is true, the adjoint of the linear operator
implemented by this method is applied instead.

See also: [`correlate`](@ref), [`fineshift!`](@ref),
[`shiftparams!`](@ref), [`correlate!`](@ref).

"""
function correlate!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    off::Int,
                    wgt::NTuple{S,T},
                    d::Int = 1,
                    adj::Bool = false,
                    opt::Int = Impl.JUMP) where {T<:AbstractFloat,N,S}
    dims, m, n = checkindices(dst, src, d)
    if adj
        fill!(dst, zero(T))
        #@assert m ≥ 1
        Impl.correlate!(Impl.Adjoint(),
                        opt, dst, src, off, wgt, d, dims, n, m)
    else
        #@assert n ≥ 1
        Impl.correlate!(Impl.Direct(),
                        opt, dst, src, off, wgt, d, dims, m, n)
    end
    return dst
end

#------------------------------------------------------------------------------
# UTILITIES

"""
    checkindices(A, B, d) -> dims, m, n

checks the indices of arguments `A` and `B` for a separable operation along
their `d`-th dimension and return `dims = size(A)`, `m = size(A,d)` and
`n = size(B,d)`.

"""
@inline function checkindices(A::AbstractArray{<:Any,N},
                              B::AbstractArray{<:Any,N},
                              d::Int) where {N}
    1 ≤ d ≤ N || throw_out_of_range_dimension_index()
    Adims = standard_size(A)
    Bdims = standard_size(B)
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
