using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = ComplexF64.(logrange(1, 1e2, 1000));

P = Legendre();
f_N = expand(P, exp);

function cauchytransforms(n)
    forward = Vector{Float64}(undef, length(x))
    inplace = Vector{Float64}(undef, length(x))
    clenshaw = Vector{Float64}(undef, length(x))

    ff = Float64.(collect(f_N.args[2][1:n]))

    forward .= abs.(FixedCauchy(n, x, ff))
    inplace .= abs.(InplaceCauchy(n, x, ff))
    clenshaw .= abs.(ClenshawCauchy(n, x, ff))

    return forward, inplace, clenshaw
end;

baseline_c = [Float64.(abs.((inv.(x₀ .- axes(P, 1)')) * f_N)) for x₀ in x];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=12,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.28inch, 6.28inch));

ax1 = Axis(
    fig[1, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{C}[\exp](x)|",
    title=L"$n = 100$",
    xscale=log10
);

forward, inplace, clenshaw = cauchytransforms(100);

lines!(ax1, real(x), baseline_c, linewidth=2);
# shift the colours
scatter!(ax1, [0], [0], visible=false);
scatter!(ax1, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax1, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax1, real(x)[60:90:end], clenshaw[60:90:end]);

ax2 = Axis(
    fig[2, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{C}[\exp](x)|",
    title=L"$n = 200$",
    xscale=log10
);

forward, inplace, clenshaw = cauchytransforms(200);

lines!(ax2, real(x), baseline_c, linewidth=2);
scatter!(ax2, [0], [0], visible=false);
scatter!(ax2, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax2, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax2, real(x)[60:90:end], clenshaw[60:90:end]);


ax3 = Axis(
    fig[3, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{C}[\exp](x)|",
    title=L"$n = 300$",
    xscale=log10
);

forward, inplace, clenshaw = cauchytransforms(300);

b = lines!(ax3, real(x), baseline_c, linewidth=2);
scatter!(ax3, [0], [0], visible=false);
f = scatter!(ax3, real(x)[1:90:end], forward[1:90:end]);
i = scatter!(ax3, real(x)[30:90:end], inplace[30:90:end]);
c = scatter!(ax3, real(x)[60:90:end], clenshaw[60:90:end]);

Legend(
    fig[4, 1],
    [b, f, i, c],
    ["baseline", "forward", "forward-inplace", "clenshaw"],
    orientation=:horizontal,
    framevisible=false,
);

fig;

save("blowup-cauchy-outside.svg", fig);