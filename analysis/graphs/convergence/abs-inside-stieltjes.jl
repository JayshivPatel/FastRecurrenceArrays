using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, GLMakie, FastGaussQuadrature;

x = range(ComplexF32(-1.0), ComplexF32(1.0), 1000);

f_g(x, z) = exp(x)/(z-x); r = 3:50;

P = Legendre(); rec_P = ClassicalOrthogonalPolynomials.recurrencecoefficients(P); f_N = expand(P, exp);

function stieltjestransform(n)
    fixed = Vector{Float32}(undef, length(x));
    ff = Float32.(collect(f_N.args[2][1:n]));

    fixed .= real(FixedStieltjes(n, x, ff));
    return fixed;
end;

baseline = [Float32.(real.(inv.(x₀ .- axes(P, 1)') * f_N)) for x₀ in x];
differences1 = Vector{Float32}(undef, length(r));
differences2 = Vector{Float32}(undef, length(r));

for i=r
    st = stieltjestransform(i);
    valid = .!(isnan.(st) .| isnan.(baseline));
    differences1[i - 2] = norm(st[valid] .- baseline[valid]);

    x_g, w_g = gausslegendre(i);
    g = [real(dot(w_g, f_g.(x_g, x₀))) for x₀ in x];
    valid = .!(isnan.(g) .| isnan.(baseline));
    differences2[i - 2] = norm(g[valid] .- baseline[valid]);
end;

fig = Figure(size = (1000, 1000));
ax1 = Axis(fig[1, 1], xlabel=L"n", ylabel="Absolute Difference", yscale=log10);
lines!(ax1, r, differences1, color=(:red, 0.9), linewidth=2);
lines!(ax1, r, differences2, color=(:blue, 0.7), linewidth=2);

fig

save("abs-inside-stieltjes.png", fig);
