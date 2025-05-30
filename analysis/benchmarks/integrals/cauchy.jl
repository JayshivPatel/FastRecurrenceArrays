using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, FastGaussQuadrature, BenchmarkTools;

x = range(ComplexF64(-10.0), ComplexF64(10.0), 500_000);

f_g(x, z) = exp(x)/(z - x);

P = Legendre();
f_N = expand(P, exp);

n_fast = 15;
n_gauss = 1000;


# only take one sample of long running tasks - precompiled

println("Cauchy Transform Baseline");
run(@benchmarkable collect(-inv.($x .- $axes(P, 1)') * $f_N))

println("Cauchy Transform FastGaussQuadrature");
output = Vector{ComplexF64}(undef, 500_000);
function gauss(x)
    x_g, w_g = gausslegendre(n_gauss);
    for (i, x₀) in enumerate(x)
        output[i] = dot(w_g, f_g.(x_g, x₀))
    end
end;
run(@benchmarkable gauss($x))

ff = Float64.(transform(P[:, 1:n_fast], exp));

println("Cauchy Transform Forward");
run(@benchmarkable FixedCauchy($n_fast, $x, $ff) samples = 100 seconds = 500)

println("Cauchy Transform Inplace");
run(@benchmarkable InplaceCauchy($n_fast, $x, $ff) samples = 100 seconds = 500)

println("Cauchy Transform Clenshaw");
run(@benchmarkable ClenshawCauchy($n_fast, $x, $ff) samples = 100 seconds = 500)
