using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/worker-scaling-10e4-recurrences-points.csv", DataFrame);

workers = df[!, "Workers"]
time = df[!, "Distributed"]

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.27inch, 2inch));
ax = Axis(
    fig[1, 1],
    xlabel="Workers",
    ylabel="Time [s]",
    xticks=(1:4)
);

d = scatterlines!(ax, workers, time);

fig

save("forward-distributed-comparison.svg", fig)