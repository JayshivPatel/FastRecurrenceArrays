using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(-2.0), ComplexF64(2.0), 1000);

P = Legendre();
f_N = expand(P, exp);

function logtransforms(n)
    forward = Vector{Float64}(undef, length(x))
    inplace = Vector{Float64}(undef, length(x))
    clenshaw = Vector{Float64}(undef, length(x))

    ff = Float64.(collect(f_N.args[2][1:n]))

    forward .= FixedLogKernel(n, x, ff)
    inplace .= InplaceLogKernel(n, x, ff)
    clenshaw .= ClenshawLogKernel(n, x, ff)

    return forward, inplace, clenshaw
end;

baseline_log = [(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];

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
    title="Logkernel stability of forward, forward-inplace and clenshaw\n\nn = 100",
);

forward, inplace, clenshaw = logtransforms(100);

lines!(ax1, real(x), baseline_log, linewidth=2);
# shift the colours
scatter!(ax1, [0], [0], visible=false);
scatter!(ax1, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax1, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax1, real(x)[60:90:end], clenshaw[60:90:end]);

ax2 = Axis(
    fig[2, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{L}[exp](z)",
    title="n = 1,000",
);

forward, inplace, clenshaw = logtransforms(1000);

lines!(ax2, real(x), baseline_log, linewidth=2);
scatter!(ax2, [0], [0], visible=false);
scatter!(ax2, real(x)[1:90:end], forward[1:90:end]);
scatter!(ax2, real(x)[30:90:end], inplace[30:90:end]);
scatter!(ax2, real(x)[60:90:end], clenshaw[60:90:end]);

ax3 = Axis(
    fig[3, 1],
    xlabel=L"z",
    ylabel=L"\mathcal{L}[exp](z)",
    title="n = 10,000",
);

forward, inplace, clenshaw = logtransforms(10_000);

b = lines!(ax3, real(x), baseline_log, linewidth=2);
scatter!(ax3, [0], [0], visible=false);
f = scatter!(ax3, real(x)[1:90:end], forward[1:90:end]);
i = scatter!(ax3, real(x)[30:90:end], inplace[30:90:end]);
c = scatter!(ax3, real(x)[60:90:end], clenshaw[60:90:end]);

Legend(
    fig[4, 1],
    [b, f, i, c],
    ["Baseline", "forward", "forward-inplace", "clenshaw"],
    orientation=:horizontal,
    framevisible=false,
);

fig

save("blowup-logkernel.svg", fig);
