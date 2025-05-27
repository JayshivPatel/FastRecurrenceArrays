using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/worker-weak-scaling.csv", DataFrame);

workers = df[!, "Workers"]
time = df[!, "Distributed"]

time = time[1] ./ time

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
    ylabel="Efficiency",
    xticks=(1:4),
    yticks=(0:0.2:1)
);

d = scatterlines!(ax, workers, ones(4), linestyle=:dot);
w = scatterlines!(ax, workers, time);

axislegend(ax, [d, w], ["ideal", "distributed"], position=:lb, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-worker-weak-scaling.svg", fig)