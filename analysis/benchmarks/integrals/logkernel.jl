using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF64(-10.0), ComplexF64(10.0), 100_000);

f_g(x, z) = log(z-x)*exp(x);

P = Legendre();
f_N = expand(P, exp);

n_fast = 20;
n_gauss = 1000;

println("Log Transform Baseline");
println(median(run(
    @benchmarkable collect(log.(abs.($x .- $axes(P, 1)')) * $f_N) samples = 100 seconds = 500
)));

ff = Float32.(collect(f_N.args[2][1:n_fast]));
println("Log Transform Forward");
println(median(run(
    @benchmarkable FixedLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500
)));

println("Log Transform Inplace");
println(median(run(
    @benchmarkable InplaceLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500
)));

println("Log Transform Clenshaw");
println(median(run(
    @benchmarkable ClenshawLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500
)));

x_g, w_g = gausslegendre(n_gauss);
println("Log Transform FastGaussQuadrature");
output = Vector{ComplexF64}(undef, 100_000);
function gauss(x)
    for (i, x₀) in enumerate(x)
        output[i] = dot(w_g, f_g.(x_g, x₀))
    end
end;
println(median(run(
    @benchmarkable gauss($x) samples = 100 seconds = 500
)));

