using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/integrals/integrals.csv", DataFrame);

cauchy = df[!, "Cauchy"];
logs = df[!, "LogKernel"];

height = vec(permutedims(hcat(cauchy, logs)));

cauchy_speedup = (cauchy[end] ./ cauchy)
logs_speedup = logs[end] ./ logs

speedups = vec(permutedims(hcat(cauchy_speedup, logs_speedup)));

table = (
    categories=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    height=height,
    groups=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
);

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.27inch, 3inch));
ax = Axis(
    fig[1, 1],
    xlabel="Time [s]",
    yticks=(1:5, ["clenshaw", "forward_inplace", "forward", "gausslegendre", "baseline"]),
    xscale=log10,
    yticklabelfont="TeX Gyre Cursor",
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
    bar_labels=speedups,
    flip_labels_at=0.85,
    color_over_background=:black,
    color_over_bar=:white,
    label_formatter=x -> string(round(x, digits=1)) * "x"
);

legend = Legend(
    fig[1, 2],
    [PolyElement(color=c2), PolyElement(color=c1)],
    ["Cauchy", "Log"],
    orientation=:vertical,
    framevisible=false,
);

fig

save("integrals.svg", fig)