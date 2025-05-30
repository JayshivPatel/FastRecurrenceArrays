using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = ComplexF64.(logrange(1, 1E5, 1000));

P = Legendre();
f_N = expand(P, exp);

function logtransforms(n)
    forward = Vector{Float64}(undef, length(x))
    inplace = Vector{Float64}(undef, length(x))
    clenshaw = Vector{Float64}(undef, length(x))

    ff = transform(P[:, 1:n], exp)

    forward .= abs.(FixedLogKernel(n, x, ff))
    inplace .= abs.(InplaceLogKernel(n, x, ff))
    clenshaw .= abs.(ClenshawLogKernel(n, x, ff))

    return forward, inplace, clenshaw
end;

baseline_log = [Float64.(abs.((log.(abs.(x₀ .- axes(P, 1)')) * f_N))) for x₀ in x];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=12,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.4inch, 8inch));

ax1 = Axis(
    fig[1, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{L}[\exp](x)|",
    title=L"$n = 3$",
    xscale=log10,
    yscale=log10
);

forward, inplace, clenshaw = logtransforms(3);

lines!(ax1, real(x), baseline_log, linewidth=2);
# shift the colours
scatter!(ax1, [0], [0], visible=false);
scatter!(ax1, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax1, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax1, real(x)[60:90:end], clenshaw[60:90:end]);

ax2 = Axis(
    fig[2, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{L}[\exp](x)|",
    title=L"$n = 4$",
    xscale=log10,
    yscale=log10
);

forward, inplace, clenshaw = logtransforms(4);

lines!(ax2, real(x), baseline_log, linewidth=2);
scatter!(ax2, [0], [0], visible=false);
scatter!(ax2, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax2, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax2, real(x)[60:90:end], clenshaw[60:90:end]);


ax3 = Axis(
    fig[3, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{L}[\exp](x)|",
    title=L"$n = 5$",
    xscale=log10,
    yscale=log10
);

forward, inplace, clenshaw = logtransforms(5);

lines!(ax3, real(x), baseline_log, linewidth=2);
scatter!(ax3, [0], [0], visible=false);
scatter!(ax3, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax3, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax3, real(x)[60:90:end], clenshaw[60:90:end]);

ax4 = Axis(
    fig[4, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{L}[\exp](x)|",
    title=L"$n = 10$",
    xscale=log10,
    yscale=log10
);

forward, inplace, clenshaw = logtransforms(10);

b = lines!(ax4, real(x), baseline_log, linewidth=2);
scatter!(ax4, [0], [0], visible=false);
f = scatter!(ax4, real(x)[1:90:end], forward[1:90:end]);
i = scatter!(ax4, real(x)[30:90:end], inplace[30:90:end]);
c = scatter!(ax4, real(x)[60:90:end], clenshaw[60:90:end]);

Legend(
    fig[5, 1],
    [b, f, i, c],
    ["baseline", "forward'", "forward_inplace", "clenshaw"],
    orientation=:horizontal,
    framevisible=false,
    labelfont="TeX Gyre Cursor",
);

fig;

save("blowup-log-outside.svg", fig);