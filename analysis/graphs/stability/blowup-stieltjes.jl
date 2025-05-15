using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, GLMakie, FastGaussQuadrature;

x = range(ComplexF32(-10.0), ComplexF32(10.0), 1000);

f_g(x, z) = exp(x)/(z-x);

P = Legendre(); rec_P = ClassicalOrthogonalPolynomials.recurrencecoefficients(P); f_N = expand(P, exp);

function stieltjestransform(n)
    fixed = Vector{Float32}(undef, length(x));
    ff = Float32.(collect(f_N.args[2][1:n]));

    fixed .= real(FixedStieltjes(n, x, ff));
    return fixed;
end;

function gaussquadrature(n)
    gauss = Vector{Float32}(undef, length(x));
    x_g, w_g = gausslegendre(n);

    gauss .= [real(dot(w_g, f_g.(x_g, x₀))) for x₀ in x];
    return gauss;
end;

baseline = [Float32.(real.((inv.(x₀ .- axes(P, 1)') * f_N))) for x₀ in x];

fig = Figure(size = (1000, 1000));

sg = SliderGrid(
    fig[4, 1],
    (
        label=L"n",
        range=3:150,
        startvalue=3,
        color_active=:grey30,
        color_active_dimmed=:grey60,
        color_inactive=:grey80,
    ),
);

n = sg.sliders[1].value;

lt = @lift stieltjestransform($n);
g = @lift gaussquadrature($n);

ax1 = Axis(fig[1, 1], xlabel=L"z", ylabel="Baseline", limits = (nothing, (-5, 15)));
ax2 = Axis(fig[2, 1], xlabel=L"z", ylabel="FastGaussQuadrature", limits = (nothing, (-5, 15)));
ax3 = Axis(fig[3, 1], xlabel=L"z", ylabel="FastRecurrenceArrays", limits = (nothing, (-5, 15)));

lines!(ax1, real(x), baseline, color=(:red, 0.9), linewidth=2);
lines!(ax2, real(x), g, color=(:blue, 0.6), linewidth=2);
lines!(ax3, real(x), lt, color=(:cyan, 0.9), linewidth=2);

fig

save("blowup-stieltjes.png", fig);