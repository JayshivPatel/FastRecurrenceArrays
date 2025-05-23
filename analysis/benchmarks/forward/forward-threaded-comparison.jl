using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/thread-scaling-10e4-recurrences-points.csv", DataFrame);

threads = df[!, "Threads"]
row = df[!, "Row-wise"]
column = df[!, "Column-wise"]

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.27inch, 2inch), padding=0);
ax = Axis(
    fig[1, 1],
    xlabel="Threads",
    ylabel="Time [s]",
    xticks=(2:2:8),
);

r = scatterlines!(ax, threads, row);
c = scatterlines!(ax, threads, column);
axislegend(ax, [r, c], ["Row-wise", "Column-wise"], position=:lt, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-threaded-comparison.svg", fig)