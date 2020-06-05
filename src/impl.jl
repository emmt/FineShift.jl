#
# impl.jl -
#
# Implementation of fast discrete correlation methods.
#
module Impl

import Base.OneTo

abstract type Operation end
struct Direct  <: Operation end # apply direct transform
struct Adjoint <: Operation end # apply adjoint transform

# Options.
#
# - Option `JUMP` specifies that, if the dimension of operation is the
#   first one (that is `d = 1`), then the innermost loop is performed along
#   the second dimension of the arrays.
#
const JUMP        =  1 # alow non-unit jump
const LOCALVALUES =  2 # store values (instead of indices) in local variables
const SPLIT       =  4 # split cases
const SIMD        =  8 # vectorization
const REFERENCE   = 16 # reference implementation

#const Indices{N} = NTuple{N,AbstractUnitRange{Int}}
const Dims{N} = NTuple{N,Int}

# If there is no version managing a specific combination of options and
# dimensions, the following will dispatch to methods using CartesianIndices
# for the indices before and after the dimension of operation.
function correlate!(dir::Operation,
                    opt::Int,
                    dst::AbstractArray{T,N},
                    src::AbstractArray{T,N},
                    off::Int,
                    wgt::NTuple{S,T},
                    d::Int,
                    dims::Dims{N},
                    m::Int,
                    n::Int) where {T<:AbstractFloat,N,S}
    correlate!(dir, opt, dst, src, off, wgt, CartesianIndices(dims[1:d-1]),
               CartesianIndices(dims[d+1:N]), m, n)
end

# Direct 1D correlation.
@generated function correlate!(::Direct,
                               opt::Int,
                               dst::AbstractArray{T,1},
                               src::AbstractArray{T,1},
                               off::Int,
                               wgt::NTuple{S,T},
                               d::Int,
                               dims::Dims{1},
                               m::Int,
                               n::Int) where {T<:AbstractFloat,S}
    @assert S ≥ 2
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    X = ntuple(k -> Symbol(:x,k), Val(S)) # neighbor values
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    Jlast = :(clamp((S-off)+i,1,n)) # index of last neighbor
    init_inds = ntuple(k -> :($(J[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))
    update_inds = assign_expr(J, (J[2:S]..., Jlast))
    Xload = ntuple(k -> :(src[$(J[k])]), Val(S))
    quote
        @assert n ≥ 1
        $(tuple_expr(W)) = wgt
        $(init_inds...)
        @inbounds @simd for i in 1:m
            $(update_inds)
            dst[i] = $(dot_expr(Xload, W))
        end
    end
end

# Adjoint 1D correlation.
@generated function correlate!(::Adjoint,
                               opt::Int,
                               dst::AbstractArray{T,1},
                               src::AbstractArray{T,1},
                               off::Int,
                               wgt::NTuple{S,T},
                               d::Int,
                               dims::Dims{1},
                               m::Int,
                               n::Int) where {T<:AbstractFloat,S}
    @assert S ≥ 2
    W = ntuple(k -> Symbol(:w,k), Val(S)) # weights
    J = ntuple(k -> Symbol(:j,k), Val(S)) # neighbor indices
    Jlast = :(clamp((S-off)+i,1,n)) # index of last neighbor
    init_inds = ntuple(k -> :($(J[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))
    update_inds = assign_expr(J, (J[2:S]..., Jlast))
    incr_dest = (:(tmp = src[i]),
                 ntuple(k -> :(dst[$(J[k])] += tmp*$(W[k])), Val(S))...)
    quote
        @assert n ≥ 1
        $(tuple_expr(W)) = wgt
        $(init_inds...)
        @inbounds for i in 1:m
            $(update_inds)
            $(incr_dest...)
        end
    end
end

# Direct 2D correlation.
@generated function correlate!(::Direct,
                               opt::Int,
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
    init_inds = assign_expr(J[2:S], P[2:S])
    update_inds = assign_expr(J, (J[2:S]..., Jlast))
    Xload1 = ntuple(k -> :(src[$(J[k]),t]), Val(S))
    Xload2 = ntuple(k -> :(src[t,$(J[k])]), Val(S))
    quote
        @assert n ≥ 1
        @inbounds o = dims[3-d]
        $(tuple_expr(W)) = wgt
        $(ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))...)
        if d > 1
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    dst[t,i] = $(dot_expr(Xload2, W))
                end
            end
        elseif (opt & JUMP) != 0
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    dst[i,t] = $(dot_expr(Xload1, W))
                end
            end
        else
            @inbounds for t in 1:o
                $(init_inds)
                @simd for i in 1:m # @simd does not help here
                    $(update_inds)
                    dst[i,t] = $(dot_expr(Xload1, W))
                end
            end
        end
    end
end

# Adjoint 2D correlation.
@generated function correlate!(::Adjoint,
                               opt::Int,
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
    init_inds = assign_expr(J[2:S], P[2:S])
    update_inds = assign_expr(J, (J[2:S]..., Jlast))
    incr_dest1 = (:(tmp = src[i,t]),
                  ntuple(k -> :(dst[$(J[k]),t] += tmp*$(W[k])), Val(S))...)
    incr_dest2 = (:(tmp = src[t,i]),
                  ntuple(k -> :(dst[t,$(J[k])] += tmp*$(W[k])), Val(S))...)
    quote
        @assert n ≥ 1
        @inbounds o = dims[3-d]
        $(tuple_expr(W)) = wgt
        $(ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), Val(S-1))...)
        if d > 1
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    $(incr_dest2...)
                end
            end
        elseif (opt & JUMP) != 0
            $(init_inds)
            @inbounds for i in 1:m
                $(update_inds)
                for t in 1:o # @simd does not help here
                    $(incr_dest1...)
                end
            end
        else
            @inbounds for t in 1:o
                $(init_inds)
                @simd for i in 1:m # @simd does not help here
                    $(update_inds)
                    $(incr_dest1...)
                end
            end
        end
    end
end

# Default version for other number of dimensions.
@generated function correlate!(::Direct,
                               opt::Int,
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
    preamble = (assign_expr(W, :wgt),
                ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), S-1)...)

    # Index of last neighbor.
    Jlast = :(clamp((S-off)+i,1,n))

    # Initialize indices before the loop along the active dimension.
    init_inds = assign_expr(J[2:S], P[2:S])

    # Update indices into the loop along the active dimension.
    update_inds = assign_expr(J, (J[2:S]..., Jlast))

    if Npre > 0
        # Operate along another dimension than the 1st one.
        Xload = ntuple(k -> :(src[ipre,$(J[k]),ipost]), Val(S))
        return quote
            $(preamble...)
            @inbounds for ipost in Ipost
                $(init_inds)
                for i in 1:m
                    $(update_inds)
                    for ipre in Ipre
                        dst[ipre,i,ipost] = $(dot_expr(Xload, W))
                    end
                end
            end
        end
    else
        # Operate along the 1st dimension.
        X = ntuple(k -> Symbol(:x,k), Val(S)) # neighbor values
        Xload = ntuple(k -> :(src[$(J[k]),ipost]), Val(S))
        init_vals = assign_expr(X[2:S], (k -> :(src[$(P[k+1]),ipost]), S-1))
        update_vals = assign_expr(X, (X[2:S]..., :(src[$(Jlast),ipost])))
        return quote
            $(preamble...)
            if (opt & LOCALVALUES) != 0
                @inbounds for ipost in Ipost
                    $(init_vals)
                    for i in 1:m
                        $(update_vals)
                        dst[i,ipost] = $(dot_expr(X, W))
                    end
                end
            else
                @inbounds for ipost in Ipost
                    $(init_inds)
                    for i in 1:m
                        $(update_inds)
                        dst[i,ipost] = $(dot_expr(Xload, W))
                    end
                end
            end
        end
    end
end

@generated function correlate!(::Adjoint,
                               opt::Int,
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
    preamble = (assign_expr(W, :wgt),
                ntuple(k -> :($(P[k+1]) = clamp($(k+1)-off,1,n)), S-1)...)

    # Initialize indices before the loop along the active dimension.
    init_inds = assign_expr(J[2:S], P[2:S])

    # Update indices into the loop along the active dimension.
    update_inds = assign_expr(J, (J[2:S]..., :(clamp((S-off)+i,1,n))))

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
# METAPROGRAMMING UTILITIES

# Types for expression operands.
const Operand = Union{Expr,Symbol}

# Tuple of operands.
const Operands = Tuple{Vararg{Operand}}

# Paris that can be converted to a tuple of operands.
const Listable = Union{Tuple{Function,Integer},
                       Tuple{Function,Val}}

"""
    assign_expr(lval, rval)

yields an expression to assign `rval` to `lval`.

"""
assign_expr(lval::Operand, rval::Operand) = Expr(:(=), lval, rval)
assign_expr(lval::Operands, rval) = assign_expr(tuple_expr(lval), rval)
assign_expr(lval::Listable, rval) = assign_expr(tuple_expr(lval...), rval)
assign_expr(lval::Operand, rval::Operands) =
    assign_expr(lval, tuple_expr(rval))
assign_expr(lval::Operand, rval::Listable) =
    assign_expr(lval, tuple_expr(rval...))

"""
    sum_expr(args)

yields an expression which is the sum of the expressions in `args`.

"""
sum_expr(args::Tuple{Vararg{Operand}}) = Expr(:call, :(+), args...)
sum_expr(args::Operand...) = sum_expr(args)

sum_expr(fn::Function, n::Integer) = sum_expr(fn, Val(Int(n)))
sum_expr(fn::Function, val::Val) = sum_expr(ntuple(fn, val))

"""
    dot_expr(X,Y)

yields an expression which is the sum of the elementwise product of
`X` and `Y`, 2 tuples of expressions and/or symbols.

"""
dot_expr(X::NTuple{N,Operand}, Y::NTuple{N,Operand}) where {N} =
    sum_expr(k -> :($(X[k])*$(Y[k])), Val(N))

# Make X into an operand.
dot_expr(X::Tuple{Function,Integer}, Y) = dot_expr((X[1], Val(Int(X[2]))), Y)
dot_expr(X::Tuple{Function,Val}, Y) = dot_expr(ntuple(X...), Y)

# Make Y into an operand.
dot_expr(X::NTuple{N,Operand}, Y::Tuple{Function,Integer}) where {N} =
    dot_expr(X, (Y[1], Val(Int(Y[2]))))
dot_expr(X::NTuple{N,Operand}, Y::Tuple{Function,Val}) where {N} =
    dot_expr(X, ntuple(Y...))

"""
    tuple_expr(args)

yields an expression which is a tuple of the expressions in `args`.

"""
tuple_expr(args::Tuple{Vararg{Operand}}) = Expr(:tuple, args...)
tuple_expr(args::Operand...) = tuple_expr(args)

tuple_expr(fn::Function, n::Integer) = tuple_expr(fn, Val(Int(n)))
tuple_expr(fn::Function, val::Val) = tuple_expr(ntuple(fn, val))

end # module
