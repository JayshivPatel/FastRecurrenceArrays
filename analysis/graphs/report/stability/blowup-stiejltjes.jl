using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF32(-2.0), ComplexF32(2.0), 1000);

P = Legendre();
f_N = expand(P, exp);

function stieltjestransform(n)
    inplace = Vector{Float32}(undef, length(x))
    ff = Float32.(collect(f_N.args[2][1:n]))

    inplace .= real(InplaceStieltjes(n, x, ff))
    return inplace
end;

baseline_st = [Float32.(real.((inv.(x₀ .- axes(P, 1)')) * f_N)) for x₀ in x];

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=12,
);

fig = Figure(size=(6.28inch, 6.28inch));

ax1 = Axis(
    fig[1, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{S}[exp](z)",
    title="Stieltjes stability of forward-inplace, n = 10",
);

b = lines!(ax1, real(x), baseline_st, linewidth=2);
i = lines!(ax1, real(x), stieltjestransform(10), linewidth=2);

ax2 = Axis(
    fig[2, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{S}[exp](z)",
    title="Stieltjes stability of forward-inplace, n = 100",
);

b = lines!(ax2, real(x), baseline_st, linewidth=2);
i = lines!(ax2, real(x), stieltjestransform(100), linewidth=2);

ax3 = Axis(
    fig[3, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{S}[exp](z)",
    title="Stieltjes stability of forward-inplace, n = 1000",
);

b = lines!(ax3, real(x), baseline_st, linewidth=2);
i = lines!(ax3, real(x), stieltjestransform(1000), linewidth=2);


Legend(
    fig[4, 1],
    [b, i],
    ["Baseline", "forward-inplace"],
    orientation=:horizontal,
    framevisible=false,
);

fig

save("blowup-stieltjes.svg", fig);
