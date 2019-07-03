module FineShiftTests

using Test, Printf, FineShift, LinearInterpolators

function FineShift.fineshift(dims::NTuple{N,Int},
                             src::AbstractArray{T,N},
                             ker::Kernel{T},
                             off::NTuple{N,Real}) where {T<:AbstractFloat,N}
   return _fineshift(dims, src, ker, off, Val(N))
end

function _fineshift(dims::NTuple{N,Int},
                    src::AbstractArray{T,N},
                    ker::Kernel{T},
                    off::NTuple{N,Real},
                    ::Val{1}) where {T<:AbstractFloat,N}
    return fineshift(dims[1], src, ker, off[1], 1)
end

function _fineshift(dims::NTuple{N,Int},
                    src::AbstractArray{T,N},
                    ker::Kernel{T},
                    off::NTuple{N,Real},
                    ::Val{D}) where {T<:AbstractFloat,N,D}
    return fineshift(dims[D], _fineshift(dims, src, ker, off, Val(D-1)),
                     ker, off[D], D)
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

end # module
