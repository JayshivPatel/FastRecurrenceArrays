using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x_in = range(ComplexF32(-1.0), ComplexF32(1.0), 1_000_000);
x_out = range(ComplexF32(1.0001), ComplexF32(10.0), 1_000_000);

f_g(x, z) = exp(x)/(z-x);

P = Legendre(); f_N = expand(P, exp);

println("Stieltjes Transform Baseline (Inside)");
println(median(
    @benchmark inv.($x_in .- $axes(P, 1)') * $f_N samples = 100 seconds = 500
));

println("Stieltjes Transform Baseline (Outside)");
println(median(
    @benchmark inv.($x_out .- $axes(P, 1)') * $f_N samples = 100 seconds = 500
));

n = 10;

ff = Float32.(collect(f_N.args[2][1:n]));
println("Stieltjes Transform FixedStieltjes (Inside)");
println(median(
    @benchmark FixedStieltjes($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform FixedStieltjes (Outside)");
println(median(
    @benchmark FixedStieltjes($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform InplaceStieltjes (Inside)");
println(median(
    @benchmark InplaceStieltjes($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform InplaceStieltjes (Outside)");
println(median(
    @benchmark InplaceStieltjes($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Threads: " * string(Threads.nthreads()));
println("Stieltjes Transform ThreadedInplaceStieltjes (Inside)");
println(median(
    @benchmark ThreadedInplaceStieltjes($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform ThreadedInplaceStieltjes (Outside)");
println(median(
    @benchmark ThreadedInplaceStieltjes($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform GPUInplaceStieltjes (Inside)");
println(median(
    @benchmark GPUInplaceStieltjes($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Stieltjes Transform GPUInplaceStieltjes (Outside)");
println(median(
    @benchmark GPUInplaceStieltjes($n, $x_out, $ff) samples = 100 seconds = 500
));

x_g, w_g = gausslegendre(n);
println("Stieltjes Transform FastGaussQuadrature (Inside)");
println(median(
    @benchmark [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x_in] samples = 100 seconds = 500
));

x_g, w_g = gausslegendre(n);
println("Stieltjes Transform FastGaussQuadrature (Outside)");
println(median(
    @benchmark [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x_out] samples = 100 seconds = 500
));