using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/point-scaling-10-recurrences.csv", DataFrame);

points = parse.(Int, replace.(df[!, "Points"], "_" => ""));
fixed = df[!, "Fixed"]
gpu = df[!, "GPU"]
column = df[!, "Column-wise (8)"]
row = df[!, "Row-wise (2)"]
distributed = df[!, "Distributed (4)"]

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=1,
);

fig = Figure(size=(6.3inch, 3inch));
ax = Axis(
    fig[1, 1],
    xlabel="Points",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10
);

f = scatterlines!(ax, points, fixed, linestyle=:dash);
g = scatterlines!(ax, points, gpu);
c = scatterlines!(ax, points, column);
r = scatterlines!(ax, points, row);
d = scatterlines!(ax, points, distributed);

Legend(
    fig[2, 1],
    [f, g, c, r, d],
    ["control", "GPU", "column-wise", "row-wise", "distributed"],
    orientation=:horizontal,
    framevisible=false
);

fig

save("forward-point-scaling.svg", fig)