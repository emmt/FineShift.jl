module FineShiftTests

using Test, FineShift, InterpolationKernels

function randu(T::Type{<:AbstractFloat}; max::Real=1, min::Real=0)
    _min = T(min)
    _max = T(max)
    return (_max - _min)*rand(T) + _min
end

function randu(T::Type{<:AbstractFloat}, dims::Dims; max::Real=1, min::Real=0)
    A = rand(T, dims)
    if max != 1 || min != 0
        a = T(max) - T(min)
        b = T(min)
        @inbounds @simd for i in eachindex(A)
            A[i] = a*A[i] + b
        end
    end
    return A
end

function FineShift.fineshift(dims::NTuple{N,Int},
                             src::AbstractArray{T,N},
                             ker::Kernel{T},
                             off::NTuple{N,Real},
                             adj::Bool = false) where {T<:AbstractFloat,N}
   return _fineshift(dims, src, ker, off, adj, Val(N))
end

function _fineshift(dims::NTuple{N,Int},
                    src::AbstractArray{T,N},
                    ker::Kernel{T},
                    off::NTuple{N,Real},
                    adj::Bool,
                    ::Val{1}) where {T<:AbstractFloat,N}
    return fineshift(dims[1], src, ker, off[1], 1, adj)
end

function _fineshift(dims::NTuple{N,Int},
                    src::AbstractArray{T,N},
                    ker::Kernel{T},
                    off::NTuple{N,Real},
                    adj::Bool,
                    ::Val{D}) where {T<:AbstractFloat,N,D}
    return fineshift(dims[D], _fineshift(dims, src, ker, off, adj, Val(D-1)),
                     ker, off[D], D, adj)
end

function vdot(x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    T = float(promote_type(Tx,Ty))
    local s::T = 0
    @assert axes(x) == axes(y)
    @inbounds for i in eachindex(x, y)
        s += T(x[i])*T(y[i])
    end
    return s
end

@testset "Fast correlation code" begin
    function splitindices(m::Int, n::Int, S::Int, l::Int)
        # This mimics the splitting of the 1D correlation code in 4 different
        # intervals.
        p = S - l
        i1 = 1 - p
        i2 = n - p
        i3 = l - 1 + n
        I1 = 1:min(i1,m)
        I2 = max(i1+1,1):min(i2,m)
        I3 = max(i2+1,1):min(i3-1,m)
        I4 = max(i3,1):m
        J = zeros(Int, S)
        if length(I2) > 0
            j = first(I2) - l
            for k in 1:S-1
                J[k+1] = clamp(j+k,1,n)
            end
        elseif length(I3) > 0
            j = first(I3) - l
            for k in 1:S-1
                J[k+1] = clamp(j+k,1,n)
            end
        end
        for i in I1
            checksplitindices(m, n, S, l, i, 1)
        end
        for i in I2
            J[1:S-1] = J[2:S]
            J[S] = p + i
            checksplitindices(m, n, S, l, i, J)
        end
        for i in I3
            J[1:S-1] = J[2:S]
            checksplitindices(m, n, S, l, i, J)
        end
        for i in I4
            checksplitindices(m, n, S, l, i, n)
        end
        return true
    end

    function checksplitindices(m::Int, n::Int, S::Int, l::Int,
                               i::Int, Jp::Vector{Int})
        for k = 1:S
            j = clamp(i-l+k,1,n)
            jp = Jp[k]
            jp == j || error("(i,k)=($i,$k) j=$j, not $jp")
        end
    end

    function checksplitindices(m::Int, n::Int, S::Int, l::Int,
                               i::Int, jp::Int)
        for k = 1:S
            j = clamp(i-l+k,1,n)
            jp == j || error("(i,k)=($i,$k) j=$j, not $jp")
        end
    end

    # For offset t and kernel support size S, l = floor(S+1+t), hence to
    # explore all cases we want t in -(S+1):(S+1), that is l in 0:2(S+1)
    # with S = 4 (as assumed in the tests) we consider l in -1:10
    @testset "(l,m,n) = ($l,$m,$n)" for m in (3,6), n in (2,4,8), l in -1:10
        @test splitindices(m, n, 4, l)
    end
end

@testset "Fine shift with zero offset" begin
    @testset "Dimensions: $dims" for dims in ((100,), (20,30), (10,20,30))
        T = Float64
        N = length(dims)
        ker = CatmullRomSpline{T}()
        src = randu(T, dims; min=-1, max=1)
        off = ntuple(i -> zero(T), N)
        dst = fineshift(dims, src, ker, off)
        @test dst == src
    end
end

@testset "Direct/adjoint fine shift" begin
    @testset "Dimensions: $dims" for dims in ((100,), (20,30), (10,20,30))
        T = Float64
        N = length(dims)
        ker = CatmullRomSpline{T}()
        xdims = dims
        ydims = ntuple(d -> (isodd(d) ? dims[d] - 1 : dims[d] + 2), N)
        x = randu(T, xdims; min=-1, max=1)
        y = randu(T, ydims; min=-1, max=1)
        off = ntuple(i -> randu(T; min=-3, max=3), N)
        @test (vdot(y, fineshift(ydims, x, ker, off, false)) ≈
               vdot(x, fineshift(xdims, y, ker, off, true )))
    end
end

@testset "Fine shift of a smooth function" begin
    function f(x::T) where {T<:AbstractFloat}
        q = T(0.03)
        r = T(1.2)
        return exp(-q*x*x)*cos(x - r)
    end
    s = 5.3 # offset in number of samples
    t = -40:0.05:40
    y1 = f.(t)
    z1 = fineshift(length(t), y1, CatmullRomSpline(), s)
    z2 = f.(t .- s*step(t))
    @test maximum(abs.(z1 .- z2)) < 1e-5
end

end # module
