using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF64(-10.0), ComplexF64(10.0), 100_000);

f_g(x, z) = exp(x)/(z - x);

P = Legendre();
f_N = expand(P, exp);

n_fast = 20;
n_gauss = 1000;

println("Cauchy Transform Baseline");
println(median(run(
    @benchmarkable collect(-inv.($x .- $axes(P, 1)') * $f_N) samples = 100 seconds = 500
)));

ff = Float32.(collect(f_N.args[2][1:n_fast]));
println("Cauchy Transform Forward");
println(median(run(
    @benchmarkable FixedCauchy($n_fast, $x, $ff) samples = 100 seconds = 500
)));

println("Cauchy Transform Inplace");
println(median(run(
    @benchmarkable InplaceCauchy($n_fast, $x, $ff) samples = 100 seconds = 500
)));

println("Cauchy Transform Clenshaw");
println(median(run(
    @benchmarkable ClenshawCauchy($n_fast, $x, $ff) samples = 100 seconds = 500
)));

x_g, w_g = gausslegendre(n_gauss);
println("Cauchy Transform FastGaussQuadrature");
output = Vector{ComplexF64}(undef, 100_000);
function gauss(x)
    for (i, x₀) in enumerate(x)
        output[i] = dot(w_g, f_g.(x_g, x₀))
    end
end;
println(median(run(
    @benchmarkable gauss($x) samples = 100 seconds = 500
)));
