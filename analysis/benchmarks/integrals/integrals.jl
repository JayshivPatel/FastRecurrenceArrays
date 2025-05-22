using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/integrals/integrals.csv", DataFrame);

cauchy = df[!, "Cauchy"];
log = df[!, "LogKernel"];

height = vec(permutedims(hcat(cauchy, log)));

table = (
    categories=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    height=height,
    groups=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
);

pt = 4 / 3;
inch = 96;

set_theme!(
    theme_latexfonts(),
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
);

fig = Figure(size=(6.27inch, 3inch));
ax = Axis(
    fig[1, 1],
    title=L"Computation time of Cauchy and Log Transforms on $10^5$ points by method",
    xlabel=L"Time [s]$$",
    yticks=(1:5, [L"clenshaw$$", L"forward-inplace$$", L"forward$$", L"gausslegendre$$", L"baseline$$"]),
    xscale=log10,
);

c1 = Makie.wong_colors()[2];
c2 = Makie.wong_colors()[1];

b = barplot!(
    table.categories,
    table.height,
    dodge=table.groups,
    color=table.groups,
    direction=:x,
    colormap=[c1, c2],
);

legend = Legend(
    fig[1, 2],
    [PolyElement(color=c2), PolyElement(color=c1)],
    ["Cauchy", "Log"],
    orientation=:vertical,
);

fig

save("integrals.svg", fig)