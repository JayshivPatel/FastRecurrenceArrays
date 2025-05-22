using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/thread-scaling-10e4-recurrences-points.csv", DataFrame);

threads = df[!, "Threads"]
row = df[!, "Row-wise"]
column = df[!, "Column-wise"]

pt = 4 / 3;
inch = 96;

set_theme!(
    theme_latexfonts(),
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
);

fig = Figure(size=(6.27inch, 3inch), padding=0);
ax = Axis(
    fig[1, 1],
    title=L"Thread scaling of forward' with $10^4$ recurrences at $10^4$ points",
    xlabel=L"Threads$$",
    ylabel=L"Time [s]$$",
    xticks=(2:2:8),
);

r = scatterlines!(ax, threads, row);
c = scatterlines!(ax, threads, column);
axislegend(ax, [r, c], [L"Row-wise$$", L"Column-wise$$"], position=:lt, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-threaded-comparison.svg", fig)