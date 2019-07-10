module FineShiftBenchmarks
using Printf
using InterpolationKernels
using BenchmarkTools
using FineShift
using FineShift.Impl: Adjoint, Direct
using FineShift.Impl: REFERENCE, JUMP, SPLIT, LOCALVALUES, SIMD
const PLOTTING = false
if PLOTTING
    using PyPlot, YPlot
    const plt = PyPlot
end
include("alt.jl")

function testalternatives()
    T = Float64
    ker = CatmullRomSpline(T)
    off = 1.7
    off,wgt = FineShift.shiftparams(ker,off)
    n1, n2 = 32, 33
    n1p, n2p = 35, 36
    arr = convert(Array{T,2}, sin.(.3 .* (1:n1)).*cos.(.4 .* (1:n2))');

    z1 = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false,Val(REFERENCE));
    z2 = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false,Val(REFERENCE));

    z1a = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false);
    z2a = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false);
    println("\nVal(0): ",extrema(z1a - z1)," ",extrema(z2a - z2))

    z1b = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false,Val(0));
    z2b = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false,Val(0));
    println("\nVal(0): ",extrema(z1b - z1)," ",extrema(z2b - z2))

    z1c = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false,Val(JUMP));
    z2c = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false,Val(JUMP));
    println("\nVal(JUMP): ",extrema(z1c - z1)," ",extrema(z2c - z2))

    z1d = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false,Val(LOCALVALUES));
    z2d = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false,Val(LOCALVALUES));
    println("\nVal(LOCALVALUES): ",extrema(z1d - z1)," ",extrema(z2d - z2))

    z1e = correlate!(Array{T,2}(undef,n1p,n2),arr,off,wgt,1,false,Val(SPLIT));
    z2e = correlate!(Array{T,2}(undef,n1,n2p),arr,off,wgt,2,false,Val(SPLIT));
    println("\nVal(SPLIT): ",extrema(z1e - z1)," ",extrema(z2e - z2))

    wrk1 = Array{T,2}(undef,n1p,n2)
    wrk2 = Array{T,2}(undef,n1,n2p)

    println("\nPackage version:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false);
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false);
    println("\nReference version:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false,Val(REFERENCE));
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false,Val(REFERENCE));
    println("\nDefault version:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false,Val(0));
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false,Val(0));
    println("\nDefault version with long jumps:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false,Val(JUMP));
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false,Val(JUMP));
    println("\nDefault version with local values:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false,Val(LOCALVALUES));
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false,Val(LOCALVALUES));
    println("\nSplit version:")
    @btime correlate!($wrk1,$arr,$off,$wgt,1,false,Val(SPLIT));
    @btime correlate!($wrk2,$arr,$off,$wgt,2,false,Val(SPLIT));
end

function test1d(T::Type{<:AbstractFloat}=Float64, len::Int=1000)
    off = T(-0.7)
    ker = CatmullRomSpline(T)
    l, w = FineShift.shiftparams(ker, off)

    x1 = rand(T,len)
    x2 = copy(x1)
    y1 = Array{T}(undef,len)
    y2 = Array{T}(undef,len)
    fineshift!(y1, x1, ker, off, 1, false)
    correlate!(y2, x2, l, w, 1, false)
    @printf("max. abs. dif. direct: %g\n", maximum(abs.(y2 - y1)))

    y1 = rand(T,len)
    y2 = copy(y1)
    x1 = Array{T}(undef,len)
    x2 = Array{T}(undef,len)
    fineshift!(x1, y1, ker, off, 1, true)
    correlate!(x2, y2, l, w, 1, true)
    @printf("max. abs. dif. direct: %g\n", maximum(abs.(x2 - x1)))


    @printf("direct  1D")
    @btime fineshift!($y1, $x1, $ker, $off, 1, false)

    @printf("adjoint 1D")
    @btime fineshift!($x1, $y1, $ker, $off, 1, true)


    @printf("correl  1D")
    @btime correlate!($y2, $x2, $l, $w)

    @printf("correl' 1D")
    @btime correlate!($x2, $y2, $l, $w, 1, true)
end

function test2d(T::Type{<:AbstractFloat}=Float64)
    m1,m2 = 33,33
    n1,n2 = 40,40
    ref = rand(T,n1,n2)
    img = Array{T}(undef,m1,m2)
    ws1 = Array{T}(undef,m1,n2)
    ws2 = Array{T}(undef,n1,m2)
    off = map(T, (0.2,-0.3))
    ker = CatmullRomSpline(T)
    l1, w1 = FineShift.shiftparams(ker, off[1])
    l2, w2 = FineShift.shiftparams(ker, off[2])

    img1 = similar(img)
    img2 = similar(img)
    fineshift!(img1, fineshift!(ws1, ref, ker, off[1], 1, false),
               ker, off[2], 2, false)
    fineshift!(img2, fineshift!(ws2, ref, ker, off[2], 2, false),
               ker, off[1], 1, false)
    @printf("(a) max. abs. dif. direct: %g\n", maximum(abs.(img2 - img1)))
    correlate!(img2, correlate!(ws1, ref, l1, w1, 1, false), l2, w2, 2, false)
    @printf("(b) max. abs. dif. direct: %g\n", maximum(abs.(img2 - img1)))
    correlate!(img1, correlate!(ws2, ref, l2, w2, 2, false), l1, w1, 1, false)
    @printf("(c) max. abs. dif. direct: %g\n", maximum(abs.(img2 - img1)))

    @printf("direct  2D (1st then 2nd dim.)")
    @btime fineshift!($img, fineshift!($ws1, $ref, $ker, $(off[1]), 1, false),
                      $ker, $(off[2]), 2, false)

    @printf("direct  2D (2nd then 1st dim.)")
    @btime fineshift!($img, fineshift!($ws2, $ref, $ker, $(off[2]), 2, false),
                      $ker, $(off[1]), 1, false)

    @printf("adjoint 2D (1st then 2nd dim.)")
    @btime fineshift!($ref, fineshift!($ws2, $img, $ker, $(off[1]), 1, true),
                      $ker, $(off[2]), 2, false)

    @printf("adjoint 2D (2nd then 1st dim.)")
    @btime fineshift!($ref, fineshift!($ws1, $img, $ker, $(off[2]), 2, true),
                      $ker, $(off[1]), 1, false)

    @printf("correlate 2D (1st then 2nd dim.)")
    @btime correlate!($img, correlate!($ws1, $ref, $l1, $w1, 1, false),
                      $l2, $w2, 2, false)

    @printf("correlate 2D (2nd then 1st dim.)")
    @btime correlate!($img, correlate!($ws2, $ref, $l2, $w2, 2, false),
                      $l1, $w1, 1, false)

end

end

FineShiftBenchmarks.testalternatives()
FineShiftBenchmarks.test1d()
FineShiftBenchmarks.test2d()
