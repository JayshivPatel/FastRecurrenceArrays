using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/thread-strong-scaling.csv", DataFrame);

threads = df[!, "Threads"]
row = df[!, "Row-wise"]
column = df[!, "Column-wise"]

row = row[1] ./ row
column = column[1] ./ column

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.27inch, 2.5inch), padding=0);
ax = Axis(
    fig[1, 1],
    xlabel="Threads",
    ylabel="Speedup",
    xticks=(2:2:8),
    yticks=(0:2),
    limits=(nothing, (0, 2.5))
);

d = scatterlines!(ax, threads, threads, linestyle=:dot);
r = scatterlines!(ax, threads, row);
c = scatterlines!(ax, threads, column);
axislegend(ax, [d, r, c], ["Ideal", "Row-wise", "Column-wise"], position=:rt, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-thread-strong-scaling.svg", fig)