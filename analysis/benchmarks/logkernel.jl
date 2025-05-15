using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF32(-10.0), ComplexF32(10.0), 100_000);

f_g(x, z) = log(z-x)*exp(x);

P = Legendre(); f_N = expand(P, exp);

println("Log Transform Baseline");
println(median(run(
    @benchmarkable collect(log.(abs.($x .- $axes(P, 1)')) * $f_N) samples = 100 seconds = 500
)));

n = 15;

ff = Float32.(collect(f_N.args[2][1:n]));
println("Log Transform FixedLogKernel");
println(median(run(
    @benchmarkable FixedLogKernel($n, $x, $ff) samples = 100 seconds = 500
)));

println("Log Transform InplaceLogKernel");
println(median(run(
    @benchmarkable InplaceLogKernel($n, $x, $ff) samples = 100 seconds = 500
)));


println("Threads: " * string(Threads.nthreads()));
println("Log Transform ThreadedInplaceLogKernel");
println(median(run(
    @benchmarkable ThreadedInplaceLogKernel($n, $x, $ff) samples = 100 seconds = 500
)));

println("Log Transform GPUInplaceLogKernel");
println(median(run(
    @benchmarkable GPUInplaceLogKernel($n, $x, $ff) samples = 100 seconds = 500
)));


x_g, w_g = gausslegendre(n);
println("Log Transform FastGaussQuadrature");
println(median(run(
    @benchmarkable [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x] samples = 100 seconds = 500
)));

