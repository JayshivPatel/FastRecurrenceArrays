using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF32(-10.0), ComplexF32(10.0), 100_000);

f_g(x, z) = exp(x) / (z - x);

P = Legendre();
f_N = expand(P, exp);

println("Cauchy Transform Baseline");
println(median(run(
    @benchmarkable collect($(-inv(2π * im)) * (inv.($x .- $axes(P, 1)') * $f_N)) samples = 100 seconds = 500
)));

n = 1_000;

ff = Float32.(collect(f_N.args[2][1:n]));
println("Cauchy Transform FixedCauchy");
println(median(run(
    @benchmarkable FixedCauchy($n, $x, $ff) samples = 100 seconds = 500
)));


println("Cauchy Transform InplaceCauchy");
println(median(run(
    @benchmarkable InplaceCauchy($n, $x, $ff) samples = 100 seconds = 500
)));


println("Threads: " * string(Threads.nthreads()));
println("Cauchy Transform ThreadedInplaceCauchy");
println(median(run(
    @benchmarkable ThreadedInplaceCauchy($n, $x, $ff) samples = 100 seconds = 500
)));

println("Cauchy Transform GPUInplaceCauchy");
println(median(run(
    @benchmarkable GPUInplaceCauchy($n, $x, $ff) samples = 100 seconds = 500
)));

x_g, w_g = gausslegendre(n);
println("Cauchy Transform FastGaussQuadrature");
println(median(run(
    @benchmarkable [$(inv(2π * im)) * dot($w_g, f_g.($x_g, x₀)) for x₀ in $x] samples = 100 seconds = 500
)));
