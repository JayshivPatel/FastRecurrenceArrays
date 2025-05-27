using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(-1.0), ComplexF64(1.0), 1000);

P = Legendre();
f_N = expand(P, exp);

function cauchytransforms(n)
    forward = Vector{Float64}(undef, length(x))
    inplace = Vector{Float64}(undef, length(x))
    clenshaw = Vector{Float64}(undef, length(x))

    ff = transform(P[:, 1:n], exp);

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

fig = Figure(size=(6.28inch, 2.5inch));

ax = Axis(
    fig[1, 1],
    xlabel=L"x",
    ylabel=L"|\mathcal{C}[\exp](x)|",
);

forward, inplace, clenshaw = cauchytransforms(100_000);

b = lines!(ax, real(x), baseline_c, linewidth=2);
# shift the colours
scatter!(ax, [0], [0], visible=false);
f = scatter!(ax, real(x)[1:90:end], forward[1:90:end]);
i = scatter!(ax, real(x)[30:90:end], inplace[30:90:end]);
c = scatter!(ax, real(x)[60:90:end], clenshaw[60:90:end]);

Legend(
    fig[2, 1],
    [b, f, i, c],
    ["baseline", "forward", "forward-inplace", "clenshaw"],
    orientation=:horizontal,
    framevisible=false,
);

fig;

save("blowup-cauchy-inside.svg", fig);