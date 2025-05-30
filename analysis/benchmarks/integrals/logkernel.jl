using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF64(-10.0), ComplexF64(10.0), 500_000);

f_g(x, z) = log(z-x)*exp(x);

P = Legendre();
f_N = expand(P, exp);

n_fast = 15;
n_gauss = 1000;

# only take one sample of long running tasks - precompiled

println("Log Transform Baseline");
run(@benchmarkable collect(log.(abs.($x .- $axes(P, 1)')) * $f_N))

println("Log Transform FastGaussQuadrature");
output = Vector{ComplexF64}(undef, 500_000);
function gauss(x)
    x_g, w_g = gausslegendre(n_gauss);
    for (i, x₀) in enumerate(x)
        output[i] = dot(w_g, f_g.(x_g, x₀))
    end
end;
run(@benchmarkable gauss($x))

ff = Float64.(transform(P[:, 1:n_fast], exp));

println("Log Transform Forward");
run(@benchmarkable FixedLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500)

println("Log Transform Inplace");
run(@benchmarkable InplaceLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500)

println("Log Transform Clenshaw");
run(@benchmarkable ClenshawLogKernel($n_fast, $x, $ff) samples = 100 seconds = 500)
