using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(-2.0), ComplexF64(2.0), 1000);

P = Legendre();
f_N = expand(P, exp);

function logtransform(n)
    inplace = Vector{Float64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    inplace .= InplaceLogKernel(n, x, ff)
    return inplace
end;

baseline_log = [Float64.(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];

pt = 4 / 3;
inch = 96;

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
    ylabel=L"\mathcal{L}[exp](z)",
    title="LogKernel stability of forward-inplace, n = 100",
);

b = lines!(ax1, real(x), baseline_log, linewidth=2);
i = lines!(ax1, real(x), logtransform(100), linewidth=2);

ax2 = Axis(
    fig[2, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{L}[exp](z)",
    title="LogKernel stability of forward-inplace, n = 1,000",
);

b = lines!(ax2, real(x), baseline_log, linewidth=2);
i = lines!(ax2, real(x), logtransform(1_000), linewidth=2);

ax3 = Axis(
    fig[3, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{L}[exp](z)",
    title="LogKernel stability of forward-inplace, n = 10,000",
);

b = lines!(ax3, real(x), baseline_log, linewidth=2);
i = lines!(ax3, real(x), logtransform(10_000), linewidth=2);


Legend(
    fig[4, 1],
    [b, i],
    ["Baseline", "forward-inplace"],
    orientation=:horizontal,
    framevisible=false,
);

fig

save("blowup-logkernel.svg", fig);
