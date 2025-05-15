using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x_in = range(ComplexF32(-1.0), ComplexF32(1.0), 1_000_000);
x_out = range(ComplexF32(1.0001), ComplexF32(10.0), 1_000_000);

f_g(x, z) = log(z-x)*exp(x);

P = Legendre(); f_N = expand(P, exp);

println("Log Transform Baseline (Inside)");
println(median(
    @benchmark log.(abs.($x_in .- $axes(P, 1)')) * $f_N samples = 100 seconds = 500
));

println("Log Transform Baseline (Outside)");
println(median(
    @benchmark log.(abs.($x_out .- $axes(P, 1)')) * $f_N samples = 100 seconds = 500
));

n = 10;

ff = Float32.(collect(f_N.args[2][1:n]));
println("Log Transform FixedLogKernel (Inside)");
println(median(
    @benchmark FixedLogKernel($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Log Transform FixedLogKernel (Outside)");
println(median(
    @benchmark FixedLogKernel($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Log Transform InplaceLogKernel (Inside)");
println(median(
    @benchmark InplaceLogKernel($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Log Transform InplaceLogKernel (Outside)");
println(median(
    @benchmark InplaceLogKernel($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Threads: " * Threads.nthreads());
println("Log Transform ThreadedInplaceLogKernel (Inside)");
println(median(
    @benchmark ThreadedInplaceLogKernel($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Log Transform ThreadedInplaceLogKernel (Outside)");
println(median(
    @benchmark ThreadedInplaceLogKernel($n, $x_out, $ff) samples = 100 seconds = 500
));

println("Log Transform GPUInplaceLogKernel (Inside)");
println(median(
    @benchmark GPUInplaceLogKernel($n, $x_in, $ff) samples = 100 seconds = 500
));

println("Log Transform GPUInplaceLogKernel (Outside)");
println(median(
    @benchmark GPUInplaceLogKernel($n, $x_out, $ff) samples = 100 seconds = 500
));

x_g, w_g = gausslegendre(n);
println("Log Transform FastGaussQuadrature (Inside)");
println(median(
    @benchmark [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x_in] samples = 100 seconds = 500
));

x_g, w_g = gausslegendre(n);
println("Log Transform FastGaussQuadrature (Outside)");
println(median(
    @benchmark [dot($w_g, f_g.($x_g, x₀)) for x₀ in $x_out] samples = 100 seconds = 500
));


