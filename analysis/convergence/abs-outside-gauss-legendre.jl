using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = [range(ComplexF64(-10.0), ComplexF64(-1.0001), 500);range(ComplexF64(1.0001), ComplexF64(10.0), 500)];

log_g(x, z) = log(z - x) * exp(x);
c_g(x, z) = exp(x) / (x - z);
r = logrange(1e1, 1e4, 100);

P = Legendre();
f_N = expand(P, exp);

baseline_log = [Float64.(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];
differencesg_log = Vector{Float64}(undef, length(r));

for (i, n) in enumerate(r)
    x_g, w_g = gausslegendre(round(Int, n))
    g = [Float64.(real(dot(w_g, log_g.(x_g, x₀)))) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesg_log[i] = 1 / num_valid * norm(g[valid] .- baseline_log[valid], 1)
end;

baseline_c = [Float64.(abs.((-inv.(x₀ .- axes(P, 1)') * f_N))) for x₀ in x];
differencesg_c = Vector{Float64}(undef, length(r));


for (i, n) in enumerate(r)
    x_g, w_g = gausslegendre(round(Int, n))
    g = [abs.(dot(w_g, c_g.(x_g, x₀))) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesg_c[i] = 1 / num_valid * norm(g[valid] .- baseline_c[valid], 1)
end;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=12,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.5inch, 5inch));

ax1 = Axis(
    fig[2, 1],
    xlabel=L"n",
    ylabel="Mean Absolute Error",
    title=L"$\mathcal{L}[\exp](x):  x \in [$-10$,$-1$) \cup ($1$, $10$] $",
    yscale=log10,
    xscale=log10,
);

g = lines!(ax1, r, differencesg_log);


ax2 = Axis(
    fig[1, 1],
    xlabel=L"n",
    ylabel="Mean Absolute Error",
    title=L"|$\mathcal{C}[\exp](x)|:  x \in [$-10$,$-1$) \cup ($1$, $10$] $",
    yscale=log10,
    xscale=log10,
);

lines!(ax2, r, differencesg_c);

fig

save("abs-outside-gauss.svg", fig);
