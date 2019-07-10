#
# alt.jl -
#
# Alternative versions of the discrete correlation for benchmarking.
#

module FineShiftAlternatives

using FineShift
using FineShift: checkindices
using FineShift.Impl: Direct, Adjoint, Operation
using FineShift.Impl: JUMP, LOCALVALUES, SPLIT, REFERENCE
using FineShift.Impl: _assign, _dot, _sum, _tuple

import FineShift: correlate!
import FineShift.Impl: _correlate!

# To distinguish the alternative versions form the ones provided by the
# main module, the option argument `opt` is an instance of `Val` not an
# `Int` and all arguments are mandatory.
function correlate!(dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    off::Int,
                    wgt::NTuple{S,T},
                    d::Int,
                    adj::Bool,
                    opt::Val) where {T<:AbstractFloat,N,S}
    dims, m, n = checkindices(dst, src, d)
    if adj
        fill!(dst, zero(T))
        #@assert m ≥ 1
        _correlate!(Adjoint(), opt, dst, src, off, wgt, d, dims, n, m)
    else
        #@assert n ≥ 1
        _correlate!(Direct(),  opt, dst, src, off, wgt, d, dims, m, n)
    end
    return dst
end

# If there is no version managing a specific combination of options and
# dimensions, the following will dispatch to methods using CartesianIndices
# for the indices before and after the dimension of operation.
function _correlate!(dir::Operation,
                     opt::Val,
                     dst::AbstractArray{T,N},
                     src::AbstractArray{T,N},
                     off::Int,
                     wgt::NTuple{S,T},
                     d::Int,
                     dims::Dims{N},
                     m::Int,
                     n::Int) where {T<:AbstractFloat,N,S}
    _correlate!(dir, opt, dst, src, off, wgt, CartesianIndices(dims[1:d-1]),
                CartesianIndices(dims[d+1:N]), m, n)
end

#------------------------------------------------------------------------------
# REFERENCE VERSIONS

@generated function _correlate!(::Direct,
                                ::Val{REFERENCE},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices,
                                m::Int,
                                n::Int) where {T,S,N,Npre}
    J = ntuple(k -> Symbol(:j,k), Val(S))
    W = ntuple(k -> Symbol(:w,k), Val(S))
    set_inds = (:(j0 = i - off),
                ntuple(k -> :($(J[k]) = clamp(j0+$k,1,n)), Val(S))...)
    if Npre > 0
        # Operate along another dimension than the 1st one.
        X = ntuple(k -> :(src[ipre,$(J[k]),ipost]), Val(S))
        return quote
            # Extract weights.
            $(_tuple(W)) = wgt
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(set_inds...)
                    for ipre in Ipre
                        dst[ipre,i,ipost] = $(_dot(X, W))
                    end
                end
            end
        end
    else
        # Operate along the 1st dimension.
        X = ntuple(k -> :(src[$(J[k]),ipost]), Val(S))
        return quote
            # Extract weights.
            $(_tuple(W)) = wgt
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(set_inds...)
                    dst[i,ipost] = $(_dot(X, W))
                end
            end
        end
    end
end

@generated function _correlate!(::Adjoint,
                                ::Val{REFERENCE},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices,
                                m::Int,
                                n::Int) where {T,S,N,Npre}
    J = ntuple(k -> Symbol(:j,k), Val(S))
    W = ntuple(k -> Symbol(:w,k), Val(S))
    set_inds = (:(j0 = i - off),
                ntuple(k -> :($(J[k]) = clamp(j0+$k,1,n)), Val(S))...)
    if Npre > 0
        # Operate along another dimension than the 1st one.
        return quote
            # Extract weights.
            $(_tuple(W)) = wgt
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(set_inds...)
                    for ipre in Ipre
                        tmp = src[ipre,i,ipost]
                        $(ntuple(
                            k -> :(dst[ipre,$(J[k]),ipost] += tmp*$(W[k])),
                            Val(S))...)
                    end
                end
            end
        end
    else
        # Operate along the 1st dimension.
        return quote
            # Extract weights.
            $(_tuple(W)) = wgt
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(set_inds...)
                    tmp = src[i,ipost]
                    $(ntuple(
                        k -> :(dst[$(J[k]),ipost] += tmp*$(W[k])),
                        Val(S))...)
                end
            end
        end
    end
end

#------------------------------------------------------------------------------
# DEFAULT VERSIONS

# Direct 2D correlation.
@generated function _correlate!(::Direct,
                                opt::Union{Val{0},Val{JUMP},
                                           Val{LOCALVALUES}},
                                dst::AbstractArray{T,2},
                                src::AbstractArray{T,2},
                                off::Int,
                                wgt::NTuple{S,T},
                                d::Int,
                                dims::Dims{2},
                                m::Int,
                                n::Int) where {T<:AbstractFloat,S}
    @assert S ≥ 2
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    X = ntuple(k -> Symbol(:x,k), Val(S)) # neighbor values
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    P = ntuple(k -> Symbol(:p,k), Val(S)) # pre-initialized indices
    Jlast = :(clamp((S-off)+i,1,n)) # index of last neighbor
    init_inds = _assign(J[2:S], P[2:S])
    update_inds = _assign(J, (J[2:S]..., Jlast))
    quote
        @assert n ≥ 1
        @inbounds o = dims[3-d]
        $(_tuple(W)) = wgt
        $(ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))...)
        if d > 1
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    dst[t,i] = $(_sum(k -> :(src[t,$(J[k])]*$(W[k])), S))
                end
            end
        elseif isa(opt, Val{JUMP})
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    dst[i,t] = $(_sum(k -> :(src[$(J[k]),t]*$(W[k])), S))
                end
            end
        elseif isa(opt, Val{LOCALVALUES})
            @inbounds for t in 1:o
                $(ntuple(k -> :($(X[k+1]) = src[$(P[k+1]),t]), Val(S-1))...)
                @simd for i in 1:m # @simd does not help here
                    $(_assign(X, (X[2:S]..., :(src[$(Jlast),t]))))
                    dst[i,t] = $(_dot(X, W))
                end
            end
        else
            @inbounds for t in 1:o
                $(init_inds)
                @simd for i in 1:m # @simd does not help here
                    $(update_inds)
                    dst[i,t] = $(_sum(k -> :(src[$(J[k]),t]*$(W[k])), S))
                end
            end
        end
    end
end

# Adjoint 2D correlation.
@generated function _correlate!(::Adjoint,
                                opt::Union{Val{0},Val{JUMP},
                                           Val{LOCALVALUES}},
                                dst::AbstractArray{T,2},
                                src::AbstractArray{T,2},
                                off::Int,
                                wgt::NTuple{S,T},
                                d::Int,
                                dims::Dims{2},
                                m::Int,
                                n::Int) where {T<:AbstractFloat,S}
    @assert S ≥ 2
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    P = ntuple(k -> Symbol(:p,k), Val(S)) # initial indices
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    Jlast = :(clamp((S-off)+i,1,n)) # index of last neighbor
    init_inds = _assign(J[2:S], P[2:S])
    update_inds = _assign(J, (J[2:S]..., Jlast))
    quote
        @assert n ≥ 1
        @inbounds o = dims[3-d]
        $(_tuple(W)) = wgt
        $(ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))...)
        if d > 1
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    tmp = src[t,i]
                    $(ntuple(k -> :(dst[t,$(J[k])] += tmp*$(W[k])),
                             Val(S))...)
                end
            end
        elseif isa(opt, Val{JUMP})
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    tmp = src[i,t]
                    $(ntuple(k -> :(dst[$(J[k]),t] += tmp*$(W[k])),
                             Val(S))...)
                end
            end
        else
            @inbounds for t in 1:o
                $(init_inds)
                @simd for i in 1:m # @simd does not help here
                    $(update_inds)
                    tmp = src[i,t]
                    $(ntuple(k -> :(dst[$(J[k]),t] += tmp*$(W[k])),
                             Val(S))...)
                end
            end
        end
    end
end

# Default version for other number of dimensions.
@generated function _correlate!(::Direct,
                                opt::Union{Val{0},Val{JUMP},
                                           Val{LOCALVALUES}},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices,
                                m::Int,
                                n::Int) where {T,S,N,Npre}
    # Symbols for local variables.
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    P = ntuple(k -> Symbol(:p,k), Val(S)) # pre-initialized indices

    # Extract weights and pre-initialize indices..
    preamble = (
        _assign(W, :wgt),
        ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), S-1)...
    )

    # Index of last neighbor.
    Jlast = :(clamp((S-off)+i,1,n))

    # Initialize indices before the loop along the active dimension.
    init_inds = _assign(J[2:S], P[2:S])

    # Update indices into the loop along the active dimension.
    update_inds = _assign(J, (J[2:S]..., Jlast))

    if Npre > 0
        # Operate along another dimension than the 1st one.
        X = ntuple(k -> :(src[ipre,$(J[k]),ipost]), Val(S))
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(init_inds)
                for i in 1:m
                    $(update_inds)
                    for ipre in Ipre
                        dst[ipre,i,ipost] = $(_dot(X, W))
                    end
                end
            end
        end
    elseif isa(opt, Val{LOCALVALUES})
        # Operate along the 1st dimension, saving the values.
        X = ntuple(k -> Symbol(:x,k), Val(S)) # neighbor values
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(_assign(X[2:S], (k -> :(src[$(P[k+1]),ipost]), S-1)))
                for i in 1:m
                    $(_assign(X, (X[2:S]..., :(src[$(Jlast),ipost]))))
                    dst[i,ipost] = $(_dot(X, W))
                end
            end
        end
    else
        # Operate along the 1st dimension, saving the indices.
        X = ntuple(k -> :(src[$(J[k]),ipost]), Val(S))
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(init_inds)
                for i in 1:m
                    $(update_inds)
                    dst[i,ipost] = $(_dot(X, W))
                end
            end
        end
    end
end

@generated function _correlate!(::Adjoint,
                                opt::Union{Val{0},Val{JUMP},
                                           Val{LOCALVALUES}},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices,
                                m::Int,
                                n::Int) where {T,S,N,Npre}
    # Symbols for local variables.
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    P = ntuple(k -> Symbol(:p,k), Val(S)) # pre-initialized indices

    # Extract weights and pre-initialize indices..
    preamble = (
        _assign(W, :wgt),
        ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), S-1)...
    )

    # Initialize indices before the loop along the active dimension.
    init_inds = _assign(J[2:S], P[2:S])

    # Update indices into the loop along the active dimension.
    update_inds = _assign(J, (J[2:S]..., :(clamp((S-off)+i,1,n))))

    if Npre > 0
        # Operate along another dimension than the 1st one.
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(init_inds)
                for i in 1:m
                    $(update_inds)
                    for ipre in Ipre
                        tmp = src[ipre,i,ipost]
                        $(ntuple(k -> :(dst[ipre,$(J[k]),ipost]
                                        += tmp*$(W[k])), S)...)
                    end
                end
            end
        end
    else
        # Operate along the 1st dimension.
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(init_inds)
                for i in 1:m
                    $(update_inds)
                    tmp = src[i,ipost]
                    $(ntuple(k -> :(dst[$(J[k]),ipost] += tmp*$(W[k])), S)...)
                end
            end
        end
    end
end

#------------------------------------------------------------------------------
# SPLIT RANGE VERSIONS

# Initialize automatic variables needed for the "split" versions.
# Assumes local variables:
# - `m` number of rows of the operator;
# - `n` number of rows of the operator;
# - `off` offset index;
# - `S` number of kernel weights;
# Defines local variable:
# - `p = S - off`
# - `i1`, `i2` and `i3` with frontier indices
# - `I1`, `I2`, `I3` and `I4` with ranges
_init_split() =
    (:(p = S - off),
     :(i1 = 1 - p),
     :(i2 = n - p),
     :(i3 = off - 1 + n),
     :(I1 = 1:min(i1,m)),
     :(I2 = max(i1+1,1):min(i2,m)),
     :(I3 = max(i2+1,1):min(i3-1,m)),
     :(I4 = max(i3,1):m))

# 1-D version which splits ranges and shifts values.
@generated function _correlate!(::Direct,
                                ::Val{SPLIT},
                                dst::AbstractVector{T},
                                src::AbstractVector{T},
                                off::Int,
                                wgt::NTuple{S,T},
                                d::Int,
                                dims::Dims{1},
                                m::Int,
                                n::Int) where {T<:AbstractFloat,S}
    # Code is for kernels of support size at least 2.
    @assert S ≥ 2

    # Symbols for neighborhood.
    W = ntuple(k -> Symbol(:w,k), Val(S))
    X = ntuple(k -> Symbol(:x,k), Val(S))

    # Initialize values x2, ..., xS before a loop.
    init_vals = ntuple(k -> :($(X[k+1]) = src[clamp(j0+$k,1,n)]), Val(S-1))

    quote
        # Split range, extract weights and compute sum of weights if needed.
        $(_init_split()...)
        $(_tuple(W)) = wgt
        local sw::T
        if length(I1) > 0 || length(I4) > 0
            sw = $(_sum(W))
        end
        @inbounds begin
            if length(I1) > 0
                q = sw*src[1]
                @simd for i in I1 # FIXME: @simd here?
                    dst[i] = q
                end
            end
            if (init = (length(I2) > 0)) == true
                j0 = first(I2) - off
                $(init_vals...)
                for i in I2
                    $(_tuple(X)) = $(_tuple(X[2:S]..., :(src[p+i])))
                    dst[i] = $(_dot(X, W))
                end
            end
            if length(I3) > 0
                if init == false
                    j0 = first(I3) - off
                    $(init_vals...)
                end
                $(X[S]) = src[n] # FIXME: not need?
                for i in I3
                    $(_assign(X[1:S-1], X[2:S]))
                    dst[i] = $(_dot(X, W))
                end
            end
            if length(I4) > 0
                q = sw*src[n]
                @simd for i in I4 # FIXME: @simd here?
                    dst[i] = q
                end
            end
        end
    end
end

# 1-D version which splits ranges and shifts indices.
@generated function _correlate!(::Adjoint,
                                ::Val{SPLIT},
                                dst::AbstractVector{T},
                                src::AbstractVector{T},
                                off::Int,
                                wgt::NTuple{S,T},
                                d::Int,
                                dims::Dims{1},
                                m::Int,
                                n::Int) where {T<:AbstractFloat,S}
    # Code is for kernels of support size at least 2.
    @assert S ≥ 2

    # Symbols for neighborhood.
    W = ntuple(k -> Symbol(:w,k), Val(S))
    J = ntuple(k -> Symbol(:j,k), Val(S))

    # Initialize indices j2, ..., jS before a loop.
    init_inds = ntuple(k -> :($(J[k+1]) = clamp(j0+$k,1,n)), Val(S-1))

    quote
        # Split range, extract weights and compute sum of weights if needed.
        $(_init_split()...)
        $(_tuple(W)) = wgt
        local sw::T
        if length(I1) > 0 || length(I4) > 0
            sw = $(_sum(W))
        end
        @inbounds begin
            if length(I1) > 0
                q = zero(T)
                for i in I1
                    q += src[i]
                end
                dst[1] += sw*q
            end
            if (init = (length(I2) > 0)) == true
                j0 = first(I2) - off
                $(init_inds...)
                for i in I2
                    $(_tuple(J)) = $(_tuple(J[2:S]..., :(p+i)))
                    tmp = src[i]
                    $(ntuple(k -> :(dst[$(J[k])] += $(W[k])*tmp), Val(S))...)
                end
            end
            if length(I3) > 0
                if init == false
                    j0 = first(I3) - off
                    $(init_inds...)
                end
                $(J[S]) = n # FIXME: not needed?
                for i in I
                    $(_assign(J[1:S-1], J[2:S]))
                    tmp = src[i]
                    $(ntuple(k -> :(dst[$(J[k])] += $(W[k])*tmp), Val(S))...)
                end
            end
            if length(I4) > 0
                q = zero(T)
                for i in I4
                    q += src[i]
                end
                dst[n] += sw*q
            end
        end
    end
end

@generated function _correlate!(::Direct,
                                opt::Val{SPLIT},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices{Npost},
                                m::Int,
                                n::Int) where {T,S,N,Npre,Npost}
    # Code is for kernels of support size at least 2.
    @assert S ≥ 2

    # The worker function `_worker!` assumes that all dimensions have been
    # checked so we can use @inbounds.
    W = ntuple(k -> Symbol(:w,k), Val(S)) # all generated weights
    J = ntuple(k -> Symbol(:j,k), Val(S)) # all generated indices
    #
    # In the returned quoted expression, the trick is to use the splat operator
    # (...) to expand the blocks of code which are tuple of expressions (like
    # `inds` below), not expressions.
    #
    # Code to compute indices of neighbors.
    inds = (:(j0 = i - off),
            ntuple(k -> :($(J[k]) = clamp(j0+$k,1,n)), Val(S))...)
    preamble = (#Expr(:meta, :inline),
                _assign(W, :wgt),
                _init_split()...)
    if Npre > 0
        # The dimension of operation may not be the first one.
        X = ntuple(k -> :(src[ipre,$(J[k]),ipost]), Val(S))
        #X = ntuple(k -> Symbol(:x,k), Val(S))
        R = ntuple(k -> Symbol(:r,k), Val(S))
        return quote
            $(preamble...)
            if length(I1) > 0 || length(I4) > 0
                # Compute sum of weights.
                sw = $(_sum(W))
            end
            mid = length(I2) > 0 || length(I3) > 0
            if mid
                j0 = first(length(I2) > 0 ? I2 : I3) - off
                $(ntuple(k -> :($(R[k+1]) = clamp(j0+$k,1,n)), Val(S-1))...)
            end
            @inbounds for ipost in Ipost
                # for i in 1:m
                #     $(inds...)
                #     @simd for ipre in Ipre
                #         dst[ipre,i,ipost] = $(_dot(X,W))
                #     end
                # end
                for i in I1
                    @simd for ipre in Ipre
                        dst[ipre,i,ipost] = sw*src[ipre,1,ipost]
                    end
                end
                if mid
                    $(ntuple(k -> :($(J[k+1]) = $(R[k+1])), Val(S-1))...)
                    for i in I2
                        $(_assign(J[1:S-1], J[2:S]))
                        $(J[S]) = p+i
                        @simd for ipre in Ipre
                            dst[ipre,i,ipost] = $(_dot(X,W))
                        end
                    end
                    for i in I3
                        $(_assign(J[1:S-1], J[2:S]))
                        @simd for ipre in Ipre
                            dst[ipre,i,ipost] = $(_dot(X,W))
                        end
                    end
                end
                for i in I4
                    @simd for ipre in Ipre
                        dst[ipre,i,ipost] = sw*src[ipre,n,ipost]
                    end
                end
            end
        end
    else
        # Code specialized when the dimension of operation is the first one.
        #X = ntuple(k -> :(src[$(J[k]),ipost]), Val(S))
        X = ntuple(k -> Symbol(:x,k), Val(S))
        init_vals = ntuple(k -> :($(X[k+1]) = src[$(J[k+1]),ipost]), Val(S-1))
        return quote
            $(preamble...)
            if length(I1) > 0 || length(I4) > 0
                # Compute sum of weights.
                sw = $(_sum(W))
            end
            if length(I2) > 0 || length(I3) > 0
                j0 = first(length(I2) > 0 ? I2 : I3) - off
                $(ntuple(k -> :($(J[k+1]) = clamp(j0+$k,1,n)), Val(S-1))...)
                @inbounds for ipost in Ipost
                    @simd for i in I1
                        dst[i,ipost] = sw*src[1,ipost]
                    end
                    $(ntuple(k -> :($(X[k+1]) = src[$(J[k+1]),ipost]),
                             Val(S-1))...)
                    @simd for i in I2
                        $(_assign(X[1:S-1], X[2:S]))
                        $(X[S]) = src[p+i,ipost]
                        dst[i,ipost] = $(_dot(X,W))
                    end
                    @simd for i in I3
                        $(_assign(X[1:S-1], X[2:S]))
                        dst[i,ipost] = $(_dot(X,W))
                    end
                    @simd for i in I4
                        dst[i,ipost] = sw*src[n,ipost]
                    end
                end
            else
                @inbounds for ipost in Ipost
                    @simd for i in I1
                        dst[i,ipost] = sw*src[1,ipost]
                    end
                    @simd for i in I4
                        dst[i,ipost] = sw*src[n,ipost]
                    end
                end
            end
        end
    end
end

@generated function _correlate!(::Adjoint,
                                opt::Val{SPLIT},
                                dst::AbstractArray{T,N},
                                src::AbstractArray{T,N},
                                off::Int,
                                wgt::NTuple{S,T},
                                Ipre::CartesianIndices{Npre},
                                Ipost::CartesianIndices{Npost},
                                m::Int,
                                n::Int) where {T,S,N,Npre,Npost}
    # Code is for kernels of support size at least 2.
    @assert S ≥ 2

    # The worker function `_worker!` assumes that all dimensions have been
    # checked so we can use @inbounds.
    W = ntuple(k -> Symbol(:w,k), Val(S)) # all generated weights
    J = ntuple(k -> Symbol(:j,k), Val(S)) # all generated indices
    #
    # In the returned quoted expression, the trick is to use the splat operator
    # (...) to expand the blocks of code which are tuple of expressions (like
    # `inds` below), not expressions.
    #
    # Code to compute indices of neighbors.
    inds = (:(j0 = i - off),
            ntuple(k -> :($(J[k]) = clamp(j0+$k,1,n)), Val(S))...)
    if Npre > 0
        # The dimension of operation may not be the first one.
        X = ntuple(k -> :(src[ipre,$(J[k]),ipost]), Val(S))
        #X = ntuple(k -> Symbol(:x,k), Val(S))
        R = ntuple(k -> Symbol(:r,k), Val(S))
        return quote
            #$(Expr(:meta, :inline))
            $(_tuple(W)) = wgt
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(inds...)
                    @simd for ipre in Ipre
                        tmp = src[ipre,i,ipost]
                        $(ntuple(k -> :(dst[ipre,$(J[k]),ipost] += $(W[k])*tmp),
                                 Val(S))...)
                    end
                end
            end
        end
    else
        # Code specialized when the dimension of operation is the first one.
        #X = ntuple(k -> :(src[$(J[k]),ipost]), Val(S))
        X = ntuple(k -> Symbol(:x,k), Val(S))
        init_vals = ntuple(k -> :($(X[k+1]) = src[$(J[k+1]),ipost]), Val(S-1))
        return quote
            #$(Expr(:meta, :inline))
            $(_init_split()...)
            $(_tuple(W)) = wgt
            if length(I1) > 0 || length(I4) > 0
                # Compute sum of weights.
                sw = $(_sum(W))
            end
            @inbounds for ipost in Ipost
                for i in 1:m
                    $(inds...)
                    tmp = src[i,ipost]
                    $(ntuple(k -> :(dst[$(J[k]),ipost] += $(W[k])*tmp),
                             Val(S))...)
                end
            end
        end
    end
end

end # module
