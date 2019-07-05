module FineShiftTests

using Test, Printf, FineShift, LazyAlgebra, LinearInterpolators

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

@testset "Fine shift with zero offset" begin
    @testset "Dimensions: $dims" for dims in ((100,), (20,30), (10,20,30))
        T = Float64
        N = length(dims)
        ker = CatmullRomSpline(T)
        src = rand(T, dims)
        off = ntuple(i -> zero(T), N)
        dst = fineshift(dims, src, ker, off)
        @test dst == src
    end
end

@testset "Direct/adjoint fine shift" begin
    @testset "Dimensions: $dims" for dims in ((100,), (20,30), (10,20,30))
        T = Float64
        N = length(dims)
        ker = CatmullRomSpline(T)
        xdims = dims
        ydims = ntuple(d -> (isodd(d) ? dims[d] - 1 : dims[d] + 2), N)
        x = randn(T, xdims)
        y = randn(T, ydims)
        off = ntuple(i -> randn(T), N)
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
