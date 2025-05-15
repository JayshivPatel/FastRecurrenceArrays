using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF32(-10.0), ComplexF32(10.0), 1_000_000);

f_g(x, z) = exp(x)/(z-x);

P = Legendre(); f_N = expand(P, exp);

println("Stieltjes Transform Baseline (Inside)");
println(median(run(
    @benchmarkable collect(inv.($x .- $axes(P, 1)') * $f_N) samples = 100 seconds = 500
)));

n = 15;

ff = Float32.(collect(f_N.args[2][1:n]));
println("Stieltjes Transform FixedStieltjes (Inside)");
println(median(run(
    @benchmarkable FixedStieltjes($n, $x, $ff) samples = 100 seconds = 500
)));


println("Stieltjes Transform InplaceStieltjes (Inside)");
println(median(run(
    @benchmarkable InplaceStieltjes($n, $x, $ff) samples = 100 seconds = 500
)));


println("Threads: " * string(Threads.nthreads()));
println("Stieltjes Transform ThreadedInplaceStieltjes (Inside)");
println(median(run(
    @benchmarkable ThreadedInplaceStieltjes($n, $x, $ff) samples = 100 seconds = 500
)));

println("Stieltjes Transform GPUInplaceStieltjes (Inside)");
println(median(run(
    @benchmarkable GPUInplaceStieltjes($n, $x, $ff) samples = 100 seconds = 500
)));

x_g, w_g = gausslegendre(n);
println("Stieltjes Transform FastGaussQuadrature (Inside)");
println(median(run(
    @benchmarkable [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x] samples = 100 seconds = 500
)));
