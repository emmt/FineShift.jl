#
# FineShift.jl -
#
# Implement direct and adjoint operator for sub-sampling separable shift of
# multi-dimensional arrays.
#
module FineShift

export
    fineshift!,
    fineshift

using LazyAlgebra
using LinearInterpolators.Kernels
import Base.OneTo

"""

```julia
fineshift!(dst, src, ker, t, d, adj=false) -> dst
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

See also [`fineshift`](@ref)

"""
function fineshift!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                    t::Real,
                    d::Int,
                    adj::Bool=false) where {T<:AbstractFloat,N,S}
    # Check arguments.
    1 ≤ d ≤ N || error("out of range dimension")
    srcinds = axes(src)
    dstinds = axes(dst)
    @inbounds for k in 1:N
        (first(srcinds[k]) == 1 && first(dstinds[k]) == 1) ||
            error("arrays have non-standard indexing")
        k == d || last(srcinds[k]) == last(dstinds[k]) ||
            throw(DimensionMismatch("dimensions mismatch"))
    end
    m = last(dstinds[d]) # number of "rows" of the operator
    n = last(srcinds[d]) # number of "columns" of the operator
    I = CartesianIndices(srcinds[1:d-1]) # leadind indices
    K = CartesianIndices(srcinds[d+1:N]) # trailing indices
    adj && fill!(dst, 0)
    return _fineshift!(dst, src, ker, T(t), I, m, n, K, adj)
end

"""

```julia
fineshift([len = size(arr, d)], arr, ker, t, d, adj=false)
```

yields a fine shift of array `arr` along the `d`-th dimension.  The shift `t`
is given in sampling units and may have a fractional part.  If `adj` is true,
the adjoint of the operator is applied instead.  The fine shift is performed by
interpolating array `arr` along its `d`-th dimension with kernel function
`ker`.  The result is an array whose `d`-th dimension is equal to `len` and
whose other dimensions are identical to those of `arr`.

See also [`fineshift!`](@ref)

"""
function fineshift(arr::AbstractArray{T,N},
                   ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                   t::Real,
                   d::Int,
                   adj::Bool=false) where {T<:AbstractFloat,N,S}
    return fineshift!(Array{T,N}(undef, size(arr)), arr, ker, t, d, adj)

end

function fineshift(len::Int,
                   arr::AbstractArray{T,N},
                   ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                   t::Real,
                   d::Int,
                   adj::Bool=false) where {T<:AbstractFloat,N,S}
    dims = ntuple(i -> (i == d ? len : size(arr, i)), Val(N))
    return fineshift!(Array{T,N}(undef, dims), arr, ker, t, d, adj)
end

getcoefs(ker::Kernel{T}, t::T) where {T<:AbstractFloat} = _getcoefs(ker, t)
getcoefs(ker::Kernel{T}, t::Real) where {T<:AbstractFloat} = getcoefs(ker, T(t))

@generated function _getcoefs(ker::Kernel{T,S}, t::T) where {T,S}
    Core.println(T)
    W = [Symbol("w",s) for s in 1:S] # all weights
    return Expr(:block,
                #Expr(:meta, :inline),
                Expr(:local, :(l::Int), [:($w::T) for w in W]...),
                :(l = floor(Int, t + S/2) + 1),
                [:($(W[s]) = ker(T(l - $s) - t)) for s in 1:S]...,
                Expr(:return, Expr(:tuple, :l, W...)))
end

# This worker function assumes that all dimensions have been checked.
#
# Typically, on my Intel Core i7-870 at 2.93GHz with S = 4 (e.g. Catmull-Rom
# spline) and small images, this code achieves 1.26 Gflops with no
# optimizations, 1.75 Gflops with @inbounds.
#
# This gives respectively 14.4 µs and 10.4 µs per image of size 36x36.
#

@generated function _fineshift!(dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                ker::Kernel{T,S,<:Union{Flat,SafeFlat}},
                                t::T,
                                I::CartesianIndices{Ni},
                                m::Int,
                                n::Int,
                                K::CartesianIndices{Nk},
                                adj::Bool=false) where {T,S,N,Ni,Nk}
    @assert S > 1
    W = [Symbol("w",s) for s in 1:S] # all generated weights
    J = [Symbol("j",s) for s in 1:S] # all generated indices

    # Generate pieces of code to: compute the interpolation coefficients,
    # compute the indices, apply the direct and apply the adjoint of the
    # operator.  For the adjoint, we assume that `dst` has been correctly
    # initialized.
    #
    # In the returned quoted expression, the trick is to use the splat operator
    # (...) to expand the blocks of code which are tuple of expressions (like
    # `coefs` and `inds` below), not expressions.
    #
    coefs = (#Expr(:local, :(l::Int), [:($w::T) for w in W]...),
             :(l = floor(Int, t + S/2) + 1),
             [:($(W[s]) = ker(T(l - $s) - t)) for s in 1:S]...)
    inds = (:(j0 = j - l),
            [:($(J[s]) = clamp(j0 + $s, 1, n)) for s in 1:S]...)
    if Ni > 0
        direct = Expr(:for, :(i = I),
                      Expr(:(=), :(dst[i, j, k]),
                           Expr(:call, :+,
                                [:(src[i, $(J[s]), k]*$(W[s])) for s in 1:S]...)))
        adjoint = Expr(:for, :(i = I),
                       Expr(:block,
                            :(tmp = src[i, j, k]),
                            [:(dst[i, $(J[s]), k] += $(W[s])*tmp) for s in 1:S]...))
    else
        direct = Expr(:(=), :(dst[j, k]),
                      Expr(:call, :+,
                           [:(src[$(J[s]), k]*$(W[s])) for s in 1:S]...))
        adjoint = Expr(:block,
                       :(tmp = src[j, k]),
                       [:(dst[$(J[s]), k] += $(W[s])*tmp) for s in 1:S]...)
    end

    quote
        $(coefs...)
        if adj
            @inbounds for k in K
                for j in 1:m
                    $(inds...)
                    $adjoint
                end
            end
        else
            @inbounds for k in K
                for j in 1:m
                    $(inds...)
                    $direct
                end
            end
        end
        return dst
    end
end

end # module
