using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, GLMakie, FastGaussQuadrature;

x = collect(range(ComplexF32(-10.0), ComplexF32(10.0), 1000));

M = length(x); T = 5;

P = Legendre(); rec_P = ClassicalOrthogonalPolynomials.recurrencecoefficients(P); f_N = expand(P, exp); 

fixed = Vector{Float32}(undef, M);
gauss = Vector{Float32}(undef, M);
baseline = collect([Float32.(real.(inv.(x₀ .- axes(P, 1)') * f_N)) for x₀ in x]);

ff = Float32.(collect(f_N.args[2][1:T])); fixed .= real.(FixedStieltjes(T, x, ff));
f_g(x, z) = exp(x)/(z-x); x_g, w_g = gausslegendre(T); gauss .= [dot(w_g, f_g.(x_g, x₀)) for x₀ in x];

fig = Figure(size = (1000, 1000));
ax = Axis(fig[1, 1], xlabel=L"z", ylabel=L"\int_{-1}^1\frac{f(t)}{z-t}dt");
lines!(ax, real(x), fixed, color=:red, linewidth=2);
lines!(ax, real(x), gauss, color=(:blue, 0.6), linewidth=2);
lines!(ax, real(x), baseline, color=:green, linewidth=2);

fig