using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/worker-scaling-10e4-recurrences-points.csv", DataFrame);

workers = df[!, "Workers"]
time = df[!, "Distributed"]

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
    title=L"Worker scaling of forward' with $10^4$ recurrences at $10^4$ points",
    xlabel=L"Workers$$",
    ylabel=L"Time [s]$$",
    xticks=(1:4)
);

d = scatterlines!(ax, workers, time);

fig

save("forward-distributed-comparison.svg", fig)